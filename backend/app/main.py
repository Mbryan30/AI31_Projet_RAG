from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config  import get_settings
from app.core.logging import setup_logging
from app.core.container import init_container
from app.api.query  import router as query_router
from app.api.health import router as health_router

setup_logging()
logger = logging.getLogger("rag.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Starting RAG backend…")
    init_container()   # loads models, builds/loads index
    logger.info("RAG backend ready ✓")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG·AI — RGPD",
        version="1.0.0",
        description="Pipeline RAG adaptatif pour la conformité RGPD",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(query_router)

    return app


app = create_app()