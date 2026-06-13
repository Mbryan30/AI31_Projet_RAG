"""
Notebook cell 4 — embedding model + ChromaDB vector store.
Persists to disk so the index survives restarts.
"""
import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger("rag.vectorstore")


def build_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def load_or_build_vectorstore(
    child_documents: list[Document] | None,
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str,
    collection_name: str = "rgpd_chunks",
) -> Chroma:
    """
    If a persisted Chroma DB already exists at `persist_dir`, load it.
    Otherwise create it from `child_documents` and persist.
    """
    persist_path = Path(persist_dir)

    if persist_path.exists() and any(persist_path.iterdir()):
        logger.info("Loading existing Chroma DB from %s", persist_dir)
        store = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_model,
            collection_name=collection_name,
        )
        logger.info("Loaded %d chunks from disk", store._collection.count())
    else:
        if not child_documents:
            raise ValueError("No child documents provided and no existing index found.")
        persist_path.mkdir(parents=True, exist_ok=True)
        logger.info("Building new Chroma DB with %d chunks", len(child_documents))
        store = Chroma.from_documents(
            documents=child_documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=str(persist_path),
        )
        logger.info("Persisted %d chunks to %s", store._collection.count(), persist_dir)

    return store