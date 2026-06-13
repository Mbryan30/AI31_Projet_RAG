"""
Notebook cells 6, 7, 8 — multi-query retrieval, parent doc recovery,
cross-encoder reranking.
"""
import logging
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag.retriever")

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Tu es un assistant spécialisé en droit RGPD.\n"
        "Reformule la question suivante en 4 variantes différentes pour améliorer\n"
        "la recherche dans une base documentaire juridique.\n"
        "Génère uniquement les 4 questions, une par ligne, sans numérotation ni tiret.\n\n"
        "Question originale : {question}\n"
        "Variantes :"
    ),
)


class RAGRetriever:
    def __init__(
        self,
        vector_store: Chroma,
        parent_index: dict[str, Document],
        llm,
        reranker_model: str,
        retriever_k: int = 6,
        multi_query_variants: int = 4,
        parent_docs_top_k: int = 8,
        reranker_top_n: int = 3,
    ):
        self.vector_store    = vector_store
        self.parent_index    = parent_index
        self.llm             = llm
        self.retriever_k     = retriever_k
        self.variants        = multi_query_variants
        self.parent_top_k    = parent_docs_top_k
        self.reranker_top_n  = reranker_top_n

        self.base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k},
        )
        logger.info("Loading reranker: %s", reranker_model)
        self.reranker = CrossEncoder(reranker_model, max_length=512)
        logger.info("RAGRetriever ready")

    # ── cell 6: multi-query ──────────────────────────────────────────

    def _generate_queries(self, question: str) -> list[str]:
        output = self.llm.invoke(MULTI_QUERY_PROMPT.format(question=question))
        text = output.content if hasattr(output, "content") else str(output)
        queries = [q.strip() for q in text.split("\n") if q.strip()]
        return queries[: self.variants]

    def _multi_query_search(self, question: str) -> list[Document]:
        queries  = self._generate_queries(question)
        all_docs: list[Document] = []
        for q in queries:
            all_docs.extend(self.base_retriever.invoke(q))
        # Deduplicate by content
        return list({doc.page_content: doc for doc in all_docs}.values())

    # ── cell 7: parent document recovery ────────────────────────────

    def retrieve_parents(self, question: str) -> list[Document]:
        child_hits = self._multi_query_search(question)
        seen: set[str] = set()
        parents: list[Document] = []

        for child in child_hits:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                doc = self.parent_index.get(pid)
                if doc:
                    parents.append(doc)
            if len(parents) >= self.parent_top_k:
                break

        return parents

    # ── cell 8: cross-encoder reranking ─────────────────────────────

    def rerank(self, question: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        pairs  = [(question, doc.page_content[:512]) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.reranker_top_n]]

    # ── metadata filter (cell 12) ────────────────────────────────────

    def retrieve_with_filter(
        self,
        question: str,
        chapitre: str | None = None,
        article:  str | None = None,
        top_k:    int = 4,
    ) -> list[Document]:
        where: dict = {}
        if chapitre and article:
            where = {"$and": [{"chapitre": chapitre}, {"article": article}]}
        elif chapitre:
            where = {"chapitre": chapitre}
        elif article:
            where = {"article": article}

        filtered = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k, "filter": where} if where else {"k": top_k},
        )
        child_hits = filtered.invoke(question)

        seen: set[str] = set()
        parents: list[Document] = []
        for child in child_hits:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                doc = self.parent_index.get(pid)
                if doc:
                    parents.append(doc)
        return parents