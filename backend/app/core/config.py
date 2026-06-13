from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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
    llm_model: str = "mistral-small-latest"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MISTRAL_API_KEY: str = ""

    # Retrieval
    retriever_k: int = 6
    multi_query_variants: int = 4
    parent_docs_top_k: int = 8
    reranker_top_n: int = 3

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
