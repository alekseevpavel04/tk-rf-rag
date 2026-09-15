from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from environment variables or .env (never from code)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.0
    llm_timeout: float = 120.0
    # false: model's chat template has no system role -> instructions are merged into the user message
    llm_system_role: bool = True

    embedding_model: str = "intfloat/multilingual-e5-small"

    vector_store: Literal["qdrant", "faiss"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tk_rf"
    faiss_dir: str = "data/index/faiss"

    raw_text_path: str = "data/raw/tk_rf.txt"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    top_k: int = 5
    min_score: float = 0.0  # hits below this cosine similarity are dropped


@lru_cache
def get_settings() -> Settings:
    return Settings()
