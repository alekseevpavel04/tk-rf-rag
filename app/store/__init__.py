from app.config import Settings
from app.store.base import SearchHit, VectorStore


def create_store(settings: Settings, name: str | None = None) -> VectorStore:
    """Pick the vector store implementation from config (VECTOR_STORE=qdrant|faiss)."""
    name = name or settings.qdrant_collection
    if settings.vector_store == "qdrant":
        from app.store.qdrant_store import QdrantStore

        return QdrantStore(url=settings.qdrant_url, collection=name)
    if settings.vector_store == "faiss":
        from app.store.faiss_store import FaissStore

        return FaissStore(directory=f"{settings.faiss_dir}/{name}")
    raise ValueError(f"unknown vector store: {settings.vector_store}")


__all__ = ["SearchHit", "VectorStore", "create_store"]
