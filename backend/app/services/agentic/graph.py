"""
Graphe LangGraph du RAG agentique multi-agents.

                  ┌──────────┐
     question ──► │ planner  │── hors_sujet ──────────────────────────────► END
                  └────┬─────┘
                       ▼
                  ┌──────────┐      ┌──────────┐
            ┌───► │researcher│ ───► │  grader  │── insuffisant ──┐
            │     └──────────┘      └────┬─────┘                 │
            │                            │ suffisant             ▼
            │                            ▼                  ┌──────────┐
            └──────────────────────── (boucle CRAG) ◄────── │ rewriter │
                                    ┌──────────┐            └──────────┘
                                    │  writer  │
                                    └────┬─────┘
                                         ▼
                                    ┌──────────┐  OK / budget épuisé
                              ┌───► │  critic  │ ─────────────────────► END
                              │     └────┬─────┘
                              │          │ à corriger
                              │          ▼
                              │     ┌──────────┐  (recherche complémentaire
                              └──── │ reviser  │   pilotée par la critique)
                                    └──────────┘

Chaque nœud est un agent spécialisé ; ils communiquent par un état partagé
(`AgentState`). Le couple critic → reviser implémente le schéma *Reflexion* :
l'agent critique sa propre réponse, formule des requêtes pour combler ses
lacunes, puis la réponse est révisée avec ce contexte enrichi.
"""
import logging
import operator
import re
import time
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.schemas.query import MessageMetrics, QueryResponse, SourceRef
from app.services.agentic import prompts as P
from app.services.agentic.tools import RGPDToolbox, doc_key, doc_label
from app.services.llm import llm_text
from app.services.pipeline import RGPD_PROMPT, _format_parents

logger = logging.getLogger("rag.agentic")


class AgentState(TypedDict, total=False):
    question: str
    route: str
    queries: list[str]              # requêtes de recherche en cours
    tried_queries: list[str]
    candidate_articles: list[int]
    candidates: list[Document]      # sortie du researcher
    scores: dict[str, float]        # score reranker par doc_key
    documents: list[Document]       # contexte validé par le grader
    missing: str
    sufficient: bool
    retrieval_loops: int
    answer: str
    critique: dict
    reflections: int
    trace: Annotated[list[str], operator.add]


def _format_candidates(docs: list[Document], max_chars: int = 1400) -> str:
    return "\n\n".join(
        f"[{i}] {doc_label(d)}\n{d.page_content[:max_chars]}" for i, d in enumerate(docs)
    )


def _format_critique(c: dict) -> str:
    lines = [f"Note : {c.get('note')}/10"]
    for key, title in (
        ("affirmations_non_fondees", "Affirmations non fondées à supprimer"),
        ("elements_manquants", "Éléments manquants"),
        ("elements_superflus", "Passages superflus à retirer"),
    ):
        items = c.get(key) or []
        if items:
            lines.append(f"{title} :\n" + "\n".join(f"- {x}" for x in items))
    return "\n".join(lines)


class AgenticRAGPipeline:
    """Orchestrateur multi-agents (LangGraph) au-dessus des outils RGPD."""

    def __init__(
        self,
        toolbox: RGPDToolbox,
        llm,
        candidates_k: int = 8,
        context_max: int = 5,
        max_retrieval_loops: int = 1,
        max_reflections: int = 2,
    ):
        self.tools = toolbox
        self.llm = llm
        self.candidates_k = candidates_k
        self.context_max = context_max
        self.max_retrieval_loops = max_retrieval_loops
        self.max_reflections = max_reflections
        self._structured_cache: dict = {}
        self.graph = self._build_graph()

    # ── utilitaire : appel structuré robuste ────────────────────────
    def _structured(self, schema, prompt_value, attempts: int = 2):
        """Appel à sortie structurée (schéma Pydantic → JSON contraint côté Ollama).
        Un modèle local peut produire un JSON invalide : on réessaie, puis on
        renvoie None (chaque agent a un comportement de repli)."""
        for i in range(attempts):
            try:
                out = self.structured_llm(schema).invoke(prompt_value)
                if out is not None:
                    return out
            except Exception as e:                   # JSON non parsable, timeout…
                logger.warning("structured output failed (%s, essai %d): %s",
                               schema.__name__, i + 1, e)
        return None

    def structured_llm(self, schema):
        if schema not in self._structured_cache:
            self._structured_cache[schema] = self.llm.with_structured_output(schema)
        return self._structured_cache[schema]

    # ── Agent 1 : planner / router ──────────────────────────────────
    def planner(self, state: AgentState) -> AgentState:
        q = state["question"]
        plan = self._structured(P.Plan, P.PLANNER_PROMPT.format_messages(question=q))
        explicit = [
            1 if m.lower() == "premier" else int(m)
            for m in re.findall(r"article\s+(\d+|premier)", q, re.IGNORECASE)
        ]
        if plan is None:
            route, subs, arts = "rgpd", [], []
        else:
            route, subs, arts = plan.route, plan.sous_questions, plan.articles_candidats
        if explicit:                                  # mention explicite = toujours RGPD
            route = "rgpd"
        arts = list(dict.fromkeys(explicit + [a for a in arts if 1 <= a <= 99]))[:6]
        # une « requête » d'un ou deux mots (ou un simple numéro) n'aide pas la recherche
        subs = [s.strip() for s in subs if len(re.findall(r"[^\W\d_]{3,}", s)) >= 3]
        queries = list(dict.fromkeys([q] + subs))[:4]
        return {
            "route": route,
            "queries": queries,
            "tried_queries": list(queries),
            "candidate_articles": arts,
            "retrieval_loops": 0,
            "reflections": 0,
            "documents": [],
            "trace": [f"planner: route={route} | requêtes={queries[1:]} | articles={arts}"],
        }

    # ── Agent 2 : researcher (outils de recherche) ──────────────────
    def researcher(self, state: AgentState) -> AgentState:
        q = state["question"]
        pool: dict[str, Document] = {}
        for query in state.get("queries", [q]):
            for d in self.tools.hybrid_search(query, k=6):
                pool.setdefault(doc_key(d), d)
        direct = []
        for n in state.get("candidate_articles", []):
            d = self.tools.get_article(n)
            if d is not None:
                direct.append(d)
                pool.setdefault(doc_key(d), d)

        ranked = self.tools.rerank(q, list(pool.values()))
        scores = {doc_key(d): s for s, d in ranked}
        top = [d for _, d in ranked[: self.candidates_k]]
        # les articles ciblés par le planner sont toujours soumis au grader
        for d in direct:
            if all(doc_key(d) != doc_key(t) for t in top):
                top.append(d)
        # les documents déjà validés lors d'une boucle précédente sont conservés
        kept = state.get("documents", [])
        kept_keys = {doc_key(d) for d in kept}
        top = [d for d in top if doc_key(d) not in kept_keys]
        return {
            "candidates": top,
            "scores": {**state.get("scores", {}), **scores},
            "trace": [f"researcher: {len(pool)} articles trouvés → {len(top)} candidats "
                      f"({', '.join(doc_label(d).split(' – ')[0] for d in top)})"],
        }

    # ── Agent 3 : grader (CRAG) ─────────────────────────────────────
    def grader(self, state: AgentState) -> AgentState:
        q = state["question"]
        cands = state.get("candidates", [])
        kept = list(state.get("documents", []))
        if not cands:
            return {"sufficient": bool(kept), "missing": "aucun document trouvé",
                    "trace": ["grader: aucun candidat"]}
        grade = self._structured(
            P.Grade,
            P.GRADER_PROMPT.format_messages(question=q, documents=_format_candidates(cands)),
        )
        if grade is None:                       # repli : on garde le top reranker
            chosen = cands[:3]
            sufficient, missing = True, ""
        else:
            idx = [i for i in dict.fromkeys(grade.pertinents) if 0 <= i < len(cands)]
            chosen = [cands[i] for i in idx]
            sufficient, missing = grade.suffisant, grade.manque
        kept.extend(chosen)
        if not kept:                            # rien de pertinent : meilleur candidat
            kept = cands[:2]
            sufficient = False
        # ordre final : score reranker décroissant, borné
        scores = state.get("scores", {})
        kept = sorted({doc_key(d): d for d in kept}.values(),
                      key=lambda d: scores.get(doc_key(d), -99), reverse=True)
        kept = kept[: self.context_max]
        return {
            "documents": kept,
            "sufficient": sufficient,
            "missing": missing,
            "trace": [f"grader: {len(chosen)}/{len(cands)} pertinents, "
                      f"suffisant={sufficient} → contexte "
                      f"[{', '.join(doc_label(d).split(' – ')[0] for d in kept)}]"],
        }

    def _after_grader(self, state: AgentState) -> str:
        if not state.get("sufficient") and state.get("retrieval_loops", 0) < self.max_retrieval_loops:
            return "rewriter"
        return "writer"

    # ── Agent 4 : rewriter ──────────────────────────────────────────
    def rewriter(self, state: AgentState) -> AgentState:
        q = state["question"]
        rw = self._structured(
            P.Rewrite,
            P.REWRITER_PROMPT.format_messages(
                question=q, tried="; ".join(state.get("tried_queries", [])),
                missing=state.get("missing") or "non précisé",
            ),
        )
        queries = (rw.requetes if rw else []) or [q]
        arts = [a for a in (rw.articles_candidats if rw else []) if 1 <= a <= 99]
        return {
            "queries": queries,
            "tried_queries": state.get("tried_queries", []) + queries,
            "candidate_articles": arts,
            "retrieval_loops": state.get("retrieval_loops", 0) + 1,
            "trace": [f"rewriter: nouvelles requêtes={queries} | articles={arts}"],
        }

    # ── Agent 5 : writer ────────────────────────────────────────────
    def writer(self, state: AgentState) -> AgentState:
        prompt = RGPD_PROMPT.format(
            context=_format_parents(state.get("documents", [])), question=state["question"]
        )
        answer = llm_text(self.llm.invoke(prompt))
        if "Réponse :" in answer:
            answer = answer[answer.rfind("Réponse :") + len("Réponse :"):].strip()
        return {"answer": answer, "trace": ["writer: premier jet rédigé"]}

    # ── Agent 6 : critic (Reflection) ───────────────────────────────
    def critic(self, state: AgentState) -> AgentState:
        c = self._structured(
            P.Critique,
            P.CRITIC_PROMPT.format_messages(
                context=_format_parents(state.get("documents", [])),
                question=state["question"], answer=state.get("answer", ""),
            ),
        )
        crit = c.model_dump() if c else {"note": 10, "fidele": True, "repond_a_la_question": True}
        try:                                    # certains modèles notent sur 20 ou 100
            note = float(crit.get("note") or 0)
            note = note / 10 if note > 20 else note / 2 if note > 10 else note
            crit["note"] = max(1, min(10, round(note)))
        except (TypeError, ValueError):
            crit["note"] = 5
        return {
            "critique": crit,
            "trace": [f"critic: note={crit.get('note')} fidèle={crit.get('fidele')} "
                      f"répond={crit.get('repond_a_la_question')} "
                      f"non_fondées={len(crit.get('affirmations_non_fondees') or [])} "
                      f"manquants={len(crit.get('elements_manquants') or [])}"],
        }

    def _after_critic(self, state: AgentState) -> str:
        c = state.get("critique", {})
        ok = (c.get("fidele") and c.get("repond_a_la_question") and (c.get("note") or 0) >= 8
              and not c.get("affirmations_non_fondees") and not c.get("elements_superflus"))
        if ok or state.get("reflections", 0) >= self.max_reflections:
            return END
        return "reviser"

    # ── Agent 7 : reviser (Reflexion : recherche + révision) ────────
    def reviser(self, state: AgentState) -> AgentState:
        q = state["question"]
        c = state.get("critique", {})
        docs = list(state.get("documents", []))
        keys = {doc_key(d) for d in docs}
        added = []
        queries = (c.get("requetes_recherche") or [])[:2]
        if queries and c.get("elements_manquants"):
            pool = {}
            for query in queries:
                for d in self.tools.hybrid_search(query, k=4):
                    if doc_key(d) not in keys:
                        pool.setdefault(doc_key(d), d)
            # on ne garde que les ajouts que le reranker juge proches de la question
            for s, d in self.tools.rerank(q, list(pool.values()), top_n=2):
                if s > 0:
                    added.append(d)
            docs = docs + added
        answer = llm_text(self.llm.invoke(P.REVISER_PROMPT.format(
            context=_format_parents(docs), question=q,
            answer=state.get("answer", ""), critique=_format_critique(c),
        )))
        if "Réponse révisée :" in answer:
            answer = answer.split("Réponse révisée :")[-1].strip()
        return {
            "answer": answer,
            "documents": docs,
            "reflections": state.get("reflections", 0) + 1,
            "trace": [f"reviser: révision #{state.get('reflections', 0) + 1}"
                      + (f" (+{len(added)} articles : "
                         f"{', '.join(doc_label(d).split(' – ')[0] for d in added)})" if added else "")],
        }

    def off_topic(self, state: AgentState) -> AgentState:
        return {"answer": P.OFF_TOPIC_ANSWER, "documents": [],
                "trace": ["off_topic: question hors RGPD"]}

    # ── construction du graphe ──────────────────────────────────────
    def _build_graph(self):
        g = StateGraph(AgentState)
        for name in ("planner", "researcher", "grader", "rewriter", "writer",
                     "critic", "reviser", "off_topic"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "planner")
        g.add_conditional_edges(
            "planner", lambda s: "off_topic" if s.get("route") == "hors_sujet" else "researcher",
            ["off_topic", "researcher"],
        )
        g.add_edge("off_topic", END)
        g.add_edge("researcher", "grader")
        g.add_conditional_edges("grader", self._after_grader, ["rewriter", "writer"])
        g.add_edge("rewriter", "researcher")
        g.add_edge("writer", "critic")
        g.add_conditional_edges("critic", self._after_critic, ["reviser", END])
        g.add_edge("reviser", "critic")
        return g.compile()

    # ── API publique ────────────────────────────────────────────────
    def run(self, question: str) -> dict:
        """Exécute le graphe → {answer, documents, contexts, trace}."""
        state = self.graph.invoke({"question": question, "trace": []},
                                  config={"recursion_limit": 40})
        docs = state.get("documents", [])
        return {
            "answer": state.get("answer", ""),
            "documents": docs,
            "contexts": [d.page_content for d in docs],
            "trace": state.get("trace", []),
            "critique": state.get("critique", {}),
        }

    def query(self, question: str) -> QueryResponse:
        t0 = time.perf_counter()
        out = self.run(question)
        scores = [self.tools.rerank(question, out["documents"])] if out["documents"] else []
        score_by_key = {doc_key(d): s for s, d in (scores[0] if scores else [])}
        sources = []
        for d in out["documents"]:
            m = d.metadata
            raw = score_by_key.get(doc_key(d), 0.0)
            sources.append(SourceRef(
                title=f"{m['article']} – {m['titre_article']} ({m['chapitre']})",
                score=max(0, min(100, int((raw + 10) / 20 * 100))),
            ))
        return QueryResponse(
            answer=out["answer"],
            sources=sources,
            metrics=MessageMetrics(
                latency=int((time.perf_counter() - t0) * 1000),
                tokens=len(out["answer"].split()),
                strategy="Agentic RAG (LangGraph multi-agents + Reflexion)",
            ),
            trace=out["trace"],
        )
