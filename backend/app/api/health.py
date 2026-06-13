import logging
from fastapi import APIRouter, Depends
from app.core.container import get_pipeline
from app.services.pipeline import RAGPipeline

logger = logging.getLogger("rag.api.health")
router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Returns 200 only when the pipeline is fully initialised."""
    return {
        "status": "ready",
        "parent_docs": len(pipeline.parent_documents),
    }