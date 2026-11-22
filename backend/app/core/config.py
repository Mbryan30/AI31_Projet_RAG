from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env à la RACINE du projet (config.py est dans backend/app/core/ → 3 niveaux au-dessus). 
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",   # tolère les clés non utilisées du .env
    )

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # CORS
    cors_origins: str = "http://localhost:5173"

    # Paths
    rgpd_html_path: str = "data/L_2016119FR.01000101.html"
    chroma_persist_dir: str = "data/chroma_db"

    # Models
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ── LLM applicatif ──
    # Fournisseur : "ollama" (local, par défaut), "mistral", "google" ou "auto"
    # (auto = Ollama si aucune clé API n'est fournie).
    llm_provider: str = "ollama"
    llm_model: str = ""                      # vide = modèle par défaut du fournisseur
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "qwen2.5:7b"
    ollama_num_ctx: int = 16384              # fenêtre de contexte (articles RGPD longs)
    ollama_keep_alive: str = "30m"
    mistral_default_model: str = "mistral-small-latest"
    google_default_model: str = "gemini-2.5-flash-lite"
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MISTRAL_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # ── Juge RAGAS (évaluation) ──
    # Modèle local plus gros que le LLM applicatif ; le même juge note la
    # baseline ET l'agentique, donc la comparaison reste équitable.
    judge_provider: str = "ollama"
    judge_model: str = "qwen2.5:14b"
    judge_num_ctx: int = 16384
    judge_max_new_tokens: int = 4096

    # ── RAG agentique (LangGraph) ──
    # Reranker multilingue : le cross-encoder ms-marco ci-dessus est entraîné
    # sur de l'anglais, ce qui pénalise un corpus 100 % français.
    agentic_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    agentic_candidates_k: int = 8      # articles candidats envoyés au grader
    agentic_context_max: int = 4       # articles max dans le contexte final
    agentic_max_retrieval_loops: int = 1   # réécritures de requête (CRAG)
    agentic_max_reflections: int = 2       # tours critique → révision (Reflexion)

    # Retrieval
    retriever_k: int = 6
    multi_query_variants: int = 4
    parent_docs_top_k: int = 8
    reranker_top_n: int = 3

    @property
    def resolved_provider(self) -> str:
        p = self.llm_provider.lower().strip()
        if p in ("ollama", "mistral", "google"):
            return p
        if self.MISTRAL_API_KEY:
            return "mistral"
        if self.GOOGLE_API_KEY:
            return "google"
        return "ollama"

    @property
    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "ollama": self.ollama_default_model,
            "google": self.google_default_model,
            "mistral": self.mistral_default_model,
        }[self.resolved_provider]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
