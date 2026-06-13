"""
POST /query   — standard JSON response  (matches frontend sendQuery())
GET  /stream  — SSE streaming response  (matches frontend streamQuery())
"""
import asyncio
import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.container import get_pipeline
from app.services.pipeline import RAGPipeline
from app.schemas.query import QueryRequest, QueryResponse, MetadataFilterRequest

logger = logging.getLogger("rag.api.query")
router = APIRouter(prefix="/query", tags=["query"])


# ── POST /query ───────────────────────────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Standard (non-streaming) RAG endpoint.
    Called by frontend lib/api.ts → sendQuery().
    """
    logger.info("query | session=%s mode=%s q=%r", body.session_id, body.mode, body.question[:60])

    # Run in thread pool to avoid blocking the event loop (CPU-heavy inference)
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.smart_rag(body.question, body.mode),
    )

    print(result)
    return result


# ── GET /stream ───────────────────────────────────────────────────────

@router.get("/stream")
async def stream(
    question:   str,
    session_id: str,
    mode:       str = "adaptive",
    sources:    str = "vectorstore",
    pipeline:   RAGPipeline = Depends(get_pipeline),
):
    """
    Server-Sent Events streaming endpoint.
    Called by frontend lib/api.ts → streamQuery().

    Events emitted:
      data: event: token  → {"token": "..."}
      data: event: done   → {"sources": [...], "metrics": {...}}
    """
    async def event_generator():
        loop   = asyncio.get_event_loop()

        # Run full pipeline in thread
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.smart_rag(question, mode),  # type: ignore[arg-type]
        )

        # Stream answer word by word
        words = result.answer.split(" ")
        for word in words:
            payload = json.dumps({"token": word + " "}, ensure_ascii=False)
            yield f"event: token\ndata: {payload}\n\n"
            await asyncio.sleep(0.02)   # ~50 words/s simulated stream

        # Done event — carries sources + metrics
        done_payload = json.dumps({
            "sources": [s.model_dump() for s in result.sources],
            "metrics": result.metrics.model_dump(),
        }, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # Nginx: disable proxy buffering
        },
    )


# ── POST /query/filter ────────────────────────────────────────────────

@router.post("/filter", response_model=list[dict])
async def filter_query(
    body: MetadataFilterRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Metadata-filtered retrieval (cell 12).
    Returns raw article metadata for debugging / advanced use.
    """
    loop  = asyncio.get_event_loop()
    docs  = await loop.run_in_executor(
        None,
        lambda: pipeline.retriever.retrieve_with_filter(
            body.question, body.chapitre, body.article, body.top_k
        ),
    )
    return [
        {
            "article":        d.metadata["article"],
            "titre_article":  d.metadata["titre_article"],
            "chapitre":       d.metadata["chapitre"],
            "titre_chapitre": d.metadata["titre_chapitre"],
            "content_preview": d.page_content[:300],
        }
        for d in docs
    ]