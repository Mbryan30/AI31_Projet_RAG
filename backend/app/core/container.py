"""
Dependency-injection container.
All heavy objects (models, vector store, pipeline) are instantiated ONCE
at startup and injected into route handlers via FastAPI Depends().
"""
import logging
from pathlib import Path

from app.core.config import get_settings
from app.services.indexing   import build_from_html
from app.services.vectorstore import build_embedding_model, load_or_build_vectorstore
from app.services.llm        import build_llm
from app.services.retriever  import RAGRetriever
from app.services.pipeline   import RAGPipeline

logger = logging.getLogger("rag.container")

# ── Singletons set at startup ────────────────────────────────────────
_pipeline:   RAGPipeline | None = None
_retriever:  RAGRetriever | None = None


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — call init_container() first")
    return _pipeline


def get_retriever() -> RAGRetriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialised — call init_container() first")
    return _retriever


def init_container() -> None:
    """Called once from app lifespan. Builds (or reloads) the entire RAG stack."""
    global _pipeline, _retriever

    settings = get_settings()
    logger.info("Initialising RAG container (env=%s)", settings.app_env)

    # 1. Parse RGPD HTML + build documents
    parent_docs, child_docs, parent_index = build_from_html(settings.rgpd_html_path)

    # 2. Embedding model
    embedding_model = build_embedding_model(settings.embedding_model)

    # 3. Vector store (load from disk if exists, else build + persist)
    vector_store = load_or_build_vectorstore(
        child_documents=child_docs,
        embedding_model=embedding_model,
        persist_dir=settings.chroma_persist_dir,
    )

    # 4. LLM
    llm = build_llm(
        api_key=settings.MISTRAL_API_KEY,
        model_name=settings.llm_model,
        max_new_tokens=settings.llm_max_new_tokens,
        temperature=settings.llm_temperature,
    )

    # 5. Retriever
    _retriever = RAGRetriever(
        vector_store=vector_store,
        parent_index=parent_index,
        llm=llm,
        reranker_model=settings.reranker_model,
        retriever_k=settings.retriever_k,
        multi_query_variants=settings.multi_query_variants,
        parent_docs_top_k=settings.parent_docs_top_k,
        reranker_top_n=settings.reranker_top_n,
    )

    # 6. Pipeline
    _pipeline = RAGPipeline(
        retriever=_retriever,
        llm=llm,
        parent_documents=parent_docs,
    )

    logger.info("RAG container ready ✓")