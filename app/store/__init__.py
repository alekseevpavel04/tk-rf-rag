import hashlib
import re

from app.config import DEFAULT_EMBEDDING_MODEL, Settings
from app.store.base import SearchHit, VectorStore


def index_name(settings: Settings) -> str:
    """Name of the collection / FAISS directory for the configured embedding model.

    Vectors of different models live in different indexes: both base and fine-tuned e5-small
    produce 384-dim vectors, so searching an index built by another model would not fail,
    it would silently return garbage. The base model keeps the old name for backward compatibility.
    """
    if settings.embedding_model == DEFAULT_EMBEDDING_MODEL:
        return settings.qdrant_collection
    slug = re.sub(r"[^a-z0-9]+", "-", settings.embedding_model.lower()).strip("-")[-48:].strip("-")
    digest = hashlib.md5(settings.embedding_model.encode()).hexdigest()[:6]
    return f"{settings.qdrant_collection}__{slug}_{digest}"


def create_store(settings: Settings, name: str | None = None) -> VectorStore:
    """Pick the vector store implementation from config (VECTOR_STORE=qdrant|faiss)."""
    name = name or index_name(settings)
    if settings.vector_store == "qdrant":
        from app.store.qdrant_store import QdrantStore

        return QdrantStore(url=settings.qdrant_url, collection=name)
    if settings.vector_store == "faiss":
        from app.store.faiss_store import FaissStore

        return FaissStore(directory=f"{settings.faiss_dir}/{name}")
    raise ValueError(f"unknown vector store: {settings.vector_store}")


__all__ = ["SearchHit", "VectorStore", "create_store", "index_name"]
