"""
Notebook cells 9 — RGPD prompt + full_rag_pipeline + smart_rag_pipeline.
Adapted to return structured QueryResponse dicts instead of calling display().
"""
import re
import time
import logging
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.services.retriever import RAGRetriever
from app.schemas.query import QueryResponse, SourceRef, MessageMetrics, RagMode

logger = logging.getLogger("rag.pipeline")

RGPD_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant expert en conformité RGPD.
Tu réponds UNIQUEMENT à partir des articles RGPD fournis dans le contexte.
Si l'information n'est pas présente, dis : "Je ne peux pas répondre avec certitude à partir des extraits RGPD fournis."
RÈGLE ABSOLUE : Ne génère jamais un article absent du contexte.

Contexte RGPD (articles complets) :
{context}

Question : {question}

Réponds en suivant EXACTEMENT cette structure :

## 1. Réponse directe
[Réponse claire et concise]

## 2. Articles RGPD concernés
[Liste des articles et chapitres du contexte]

## 3. Explication détaillée
[Ce que disent les articles pertinents]

## 4. Points de vigilance
[Base légale, consentement, minimisation, données sensibles]

## 5. Recommandation pratique
[Conseil concret et actionnable]

⚠️ Cette réponse est informative et ne constitue pas un avis juridique.

Réponse :
"""
)

# Map mode → strategy label (for metrics)
MODE_STRATEGY: dict[str, str] = {
    "adaptive": "Smart Router",
    "crag":     "CRAG + RRF",
    "hyde":     "HyDE + Rerank",
    "selfrag":  "Self-RAG",
}


def _format_parents(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(
            f"{'='*60}\n{m['chapitre']} – {m['titre_chapitre']}\n"
            f"{m['article']} – {m['titre_article']}\n{'='*60}\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def _docs_to_sources(docs: list[Document], scores: list[float] | None = None) -> list[SourceRef]:
    """Convert reranked docs to SourceRef list with 0-100 scores."""
    sources = []
    for i, doc in enumerate(docs):
        m = doc.metadata
        label = f"{m['article']} – {m['titre_article']} ({m['chapitre']})"
        # Normalize reranker scores to 0-100 (scores are logits, typically -10 to +10)
        if scores:
            raw = scores[i]
            score = max(0, min(100, int((raw + 10) / 20 * 100)))
        else:
            score = max(0, 100 - i * 8)   # fallback: decay by rank
        sources.append(SourceRef(title=label, score=score))
    return sources


class RAGPipeline:
    def __init__(self, retriever: RAGRetriever, llm, parent_documents: list[Document]):
        self.retriever        = retriever
        self.llm              = llm
        self.parent_documents = parent_documents

    def full_rag(self, question: str, mode: RagMode = "adaptive") -> QueryResponse:
        t0 = time.perf_counter()

        parents   = self.retriever.retrieve_parents(question)
        reranked  = self.retriever.rerank(question, parents)

        context   = _format_parents(reranked)
        prompt    = RGPD_PROMPT.format(context=context, question=question)
        result = self.llm.invoke(prompt)
        raw: str = result.content if hasattr(result, "content") else str(result)

        # Strip echoed prompt/marker
        marker = "Réponse :"
        if marker in raw:
            raw = raw[raw.rfind(marker) + len(marker):].strip()

        latency = int((time.perf_counter() - t0) * 1000)

        # Approximate token count
        tokens  = len(raw.split())

        sources = _docs_to_sources(reranked)

        return QueryResponse(
            answer=raw,
            sources=sources,
            metrics=MessageMetrics(
                latency=latency,
                tokens=tokens,
                strategy=MODE_STRATEGY.get(mode, "RAG"),
            ),
        )

    def smart_rag(self, question: str, mode: RagMode = "adaptive") -> QueryResponse:
        """
        Entry point matching notebook's smart_rag_pipeline():
        - Article mention  → direct metadata lookup (no LLM)
        - Chapter mention  → chapter-level lookup
        - Otherwise        → full semantic RAG
        """
        t0 = time.perf_counter()

        # ── 1. Direct article lookup ─────────────────────────────────
        match = re.search(r"article\s+(\d+|premier)", question, re.IGNORECASE)
        if match:
            numero = match.group(1).capitalize()
            key    = f"Article {numero}"
            found  = [
                d for d in self.parent_documents
                if d.metadata["article"].lower() == key.lower()
            ]
            if found:
                d, m = found[0], found[0].metadata
                answer = (
                    f"## 1. Réponse directe\n"
                    f"L'{key} se trouve dans le {m['chapitre']} – {m['titre_chapitre']}.\n\n"
                    f"## 2. Article concerné\n{m['article']} – {m['titre_article']}\n\n"
                    f"## 3. Contenu complet\n{d.page_content}"
                )
                return QueryResponse(
                    answer=answer,
                    sources=[SourceRef(
                        title=f"{m['article']} – {m['titre_article']} ({m['chapitre']})",
                        score=100,
                    )],
                    metrics=MessageMetrics(
                        latency=int((time.perf_counter() - t0) * 1000),
                        tokens=len(answer.split()),
                        strategy="Metadata Lookup",
                    ),
                )
            return QueryResponse(
                answer=f"{key} n'existe pas dans le RGPD (99 articles au total).",
                sources=[],
                metrics=MessageMetrics(
                    latency=int((time.perf_counter() - t0) * 1000),
                    tokens=10,
                    strategy="Metadata Lookup",
                ),
            )

        # ── 2. Chapter title lookup ──────────────────────────────────
        chapter_hit = next(
            (
                d for d in self.parent_documents
                if d.metadata["titre_chapitre"].lower() in question.lower()
            ),
            None,
        )
        if chapter_hit:
            m = chapter_hit.metadata
            arts = [
                d for d in self.parent_documents
                if d.metadata["chapitre"] == m["chapitre"]
            ]
            answer = (
                f"## 1. Réponse directe\n"
                f"Le chapitre qui traite de « {m['titre_chapitre']} » "
                f"est le **{m['chapitre']}**.\n\n"
                f"## 2. Articles concernés\n"
                + "\n".join(
                    f"- {a.metadata['article']} – {a.metadata['titre_article']}"
                    for a in arts
                )
            )
            sources = [
                SourceRef(
                    title=f"{a.metadata['article']} ({m['chapitre']})",
                    score=95,
                )
                for a in arts[:5]
            ]
            return QueryResponse(
                answer=answer,
                sources=sources,
                metrics=MessageMetrics(
                    latency=int((time.perf_counter() - t0) * 1000),
                    tokens=len(answer.split()),
                    strategy="Chapter Lookup",
                ),
            )

        # ── 3. Full semantic RAG ─────────────────────────────────────
        return self.full_rag(question, mode)