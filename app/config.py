from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Base model and the model fine-tuned on Russian law in github.com/alekseevpavel04/ru-law-retrieval
# (same architecture and the same "query: " / "passage: " prefixes, so they are interchangeable).
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
FINETUNED_EMBEDDING_MODEL = "alekseevpavel04/multilingual-e5-small-ru-law"


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

    # HF id or a local path. Each model gets its own index (see app/store.index_name): switching the
    # model requires POST /ingest, vectors of different models are never mixed.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    vector_store: Literal["qdrant", "faiss"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tk_rf"
    faiss_dir: str = "data/index/faiss"

    raw_text_path: str = "data/raw/tk_rf.txt"
    chunk_size: int = 500  # chosen by eval: see eval/results.md
    chunk_overlap: int = 75

    top_k: int = 5
    # hits below this cosine similarity are dropped; the scale depends on the embedding model
    # (the fine-tuned model gives lower absolute cosines), so a threshold must be re-tuned per model
    min_score: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
