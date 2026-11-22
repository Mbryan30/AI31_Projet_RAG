"""
Outils de recherche mis à disposition des agents.

  • get_article(n)       — accès direct à un article par son numéro (lookup métadonnées)
  • dense_search(q)      — recherche sémantique sur les petits chunks → articles parents
  • bm25_search(q)       — recherche lexicale BM25 sur les articles complets (titre inclus)
  • hybrid_search(q)     — fusion dense + BM25 par Reciprocal Rank Fusion (RRF)
  • rerank(q, docs)      — reranking cross-encoder multilingue

Pourquoi l'hybride ? Dans le pipeline de base, seuls les chunks *sans titre*
d'article sont vectorisés : une question qui reprend presque mot pour mot le
titre d'un article (« principes relatifs au traitement… » = titre de l'article 5)
peut le rater. BM25 sur l'article complet (titre compris) corrige ce point.
"""
import logging
import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag.agentic.tools")

_STOPWORDS = set("""
a à au aux avec ce ces cette dans de des du elle en et eux il ils je la le les leur
leurs lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que
qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y d l s n c j m t
est sont été être ont a quel quelle quels quelles quoi comment quand peut doit doivent
peuvent cas lorsque sein ainsi entre selon tout tous toute toutes autre autres
""".split())


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize(text))
    out = []
    for t in tokens:
        if t in _STOPWORDS or len(t) < 2:
            continue
        # racinisation très légère (pluriels / féminins fréquents)
        for suf in ("ements", "ement", "ations", "ation", "es", "s", "e"):
            if len(t) > len(suf) + 3 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


def article_number(label: str) -> int | None:
    """'Article 5' → 5 ; 'Article premier' → 1."""
    m = re.search(r"article\s+(\d+|premier)", label, re.IGNORECASE)
    if not m:
        return None
    return 1 if m.group(1).lower() == "premier" else int(m.group(1))


def doc_key(doc: Document) -> str:
    return doc.metadata.get("parent_id") or doc.page_content[:80]


def doc_label(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('article', '?')} – {m.get('titre_article', '')}".strip(" –")


class RGPDToolbox:
    def __init__(
        self,
        vector_store,
        parent_documents: list[Document],
        parent_index: dict[str, Document],
        reranker_model: str,
        dense_k: int = 12,
    ):
        self.parent_documents = parent_documents
        self.parent_index = parent_index
        self.dense = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": dense_k}
        )
        self.by_number: dict[int, Document] = {}
        for d in parent_documents:
            n = article_number(d.metadata.get("article", ""))
            if n is not None and n not in self.by_number:
                self.by_number[n] = d

        self._bm25_docs = parent_documents
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in parent_documents])

        logger.info("Loading agentic reranker: %s", reranker_model)
        self.reranker = CrossEncoder(reranker_model, max_length=512)
        logger.info("RGPDToolbox ready (%d articles)", len(self.by_number))

    # ── outils élémentaires ─────────────────────────────────────────
    def get_article(self, n: int) -> Document | None:
        return self.by_number.get(int(n))

    def dense_search(self, query: str, k: int = 6) -> list[Document]:
        seen: set[str] = set()
        out: list[Document] = []
        for child in self.dense.invoke(query):
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen and pid in self.parent_index:
                seen.add(pid)
                out.append(self.parent_index[pid])
            if len(out) >= k:
                break
        return out

    def bm25_search(self, query: str, k: int = 6) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._bm25_docs[i] for i in ranked[:k] if scores[i] > 0]

    def hybrid_search(self, query: str, k: int = 6, rrf_k: int = 60) -> list[Document]:
        fused: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for ranking in (self.dense_search(query, k * 2), self.bm25_search(query, k * 2)):
            for rank, d in enumerate(ranking):
                key = doc_key(d)
                docs[key] = d
                fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [docs[key] for key in best]

    def rerank(self, question: str, docs: list[Document], top_n: int | None = None
               ) -> list[tuple[float, Document]]:
        if not docs:
            return []
        pairs = [(question, d.page_content[:1500]) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip((float(s) for s in scores), docs), key=lambda x: x[0], reverse=True)
        return ranked[:top_n] if top_n else ranked
