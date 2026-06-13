from pydantic import BaseModel, Field
from typing import Literal


# ── Enums matching frontend types/index.ts ───────────────────────────

RagMode   = Literal["adaptive", "crag", "hyde", "selfrag"]
SourceId  = Literal["vectorstore", "postgres", "neo4j", "web"]


# ── Request ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str      = Field(..., min_length=1, max_length=2000)
    session_id: str      = Field(..., description="Frontend session UUID")
    mode:       RagMode  = "adaptive"
    sources:    list[SourceId] = ["vectorstore"]


class MetadataFilterRequest(BaseModel):
    question: str
    chapitre: str | None = None
    article:  str | None = None
    top_k:    int        = Field(default=4, ge=1, le=20)


# ── Response ─────────────────────────────────────────────────────────

class SourceRef(BaseModel):
    """Maps 1-to-1 with frontend SourceRef interface."""
    title: str
    score: int = Field(..., ge=0, le=100, description="Relevance score 0-100")
    url:   str | None = None


class MessageMetrics(BaseModel):
    """Maps 1-to-1 with frontend MessageMetrics interface."""
    latency:  int    = Field(..., description="End-to-end latency in ms")
    tokens:   int    = Field(..., description="Tokens generated")
    strategy: str    = Field(..., description="RAG strategy used")


class QueryResponse(BaseModel):
    """Root response — matches frontend QueryResponse interface exactly."""
    answer:  str
    sources: list[SourceRef]
    metrics: MessageMetrics


# ── SSE streaming token ───────────────────────────────────────────────

class StreamToken(BaseModel):
    token: str


class StreamDone(BaseModel):
    sources: list[SourceRef]
    metrics: MessageMetrics


# ── Indexing ─────────────────────────────────────────────────────────

class IndexingStatus(BaseModel):
    status:   Literal["pending", "running", "done", "error"]
    chapters: int = 0
    parents:  int = 0
    children: int = 0
    message:  str = ""