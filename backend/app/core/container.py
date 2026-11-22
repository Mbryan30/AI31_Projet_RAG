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
from app.services.llm        import build_llm_from_settings
from app.services.retriever  import RAGRetriever
from app.services.pipeline   import RAGPipeline
from app.services.agentic.tools import RGPDToolbox
from app.services.agentic.graph import AgenticRAGPipeline

logger = logging.getLogger("rag.container")

# ── Singletons set at startup ────────────────────────────────────────
_pipeline:   RAGPipeline | None = None
_retriever:  RAGRetriever | None = None
_agentic:    AgenticRAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — call init_container() first")
    return _pipeline


def get_retriever() -> RAGRetriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialised — call init_container() first")
    return _retriever


def get_agentic() -> AgenticRAGPipeline:
    if _agentic is None:
        raise RuntimeError("Agentic pipeline not initialised — call init_container() first")
    return _agentic


def init_container(with_agentic: bool = True) -> None:
    """Called once from app lifespan. Builds (or reloads) the entire RAG stack."""
    global _pipeline, _retriever, _agentic

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

    # 4. LLM (Mistral ou Gemini selon les clés présentes dans .env)
    llm = build_llm_from_settings(settings)
    logger.info("LLM: %s / %s", settings.resolved_provider, settings.resolved_model)

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

    # 6. RAG agentique multi-agents (LangGraph)
    if with_agentic:
        toolbox = RGPDToolbox(
            vector_store=vector_store,
            parent_documents=parent_docs,
            parent_index=parent_index,
            reranker_model=settings.agentic_reranker_model,
        )
        _agentic = AgenticRAGPipeline(
            toolbox=toolbox,
            llm=llm,
            candidates_k=settings.agentic_candidates_k,
            context_max=settings.agentic_context_max,
            max_retrieval_loops=settings.agentic_max_retrieval_loops,
            max_reflections=settings.agentic_max_reflections,
        )

    # 7. Pipeline (mode "agentic" délégué au graphe LangGraph)
    _pipeline = RAGPipeline(
        retriever=_retriever,
        llm=llm,
        parent_documents=parent_docs,
        agentic=_agentic,
    )

    logger.info("RAG container ready ✓")