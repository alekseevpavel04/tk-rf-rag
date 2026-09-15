from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.ingest.chunker import Chunk


@dataclass
class SearchHit:
    score: float  # cosine similarity
    payload: dict  # article, title, chapter, chapter_title, position, text


class VectorStore(ABC):
    """Minimal interface the RAG pipeline needs from a vector database."""

    name: str

    @abstractmethod
    def recreate(self, dim: int) -> None:
        """Drop existing data and prepare an empty index of given dimension."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Store chunks with their (L2-normalized) vectors."""

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int, chapter: str | None = None) -> list[SearchHit]:
        """Return top_k hits by cosine similarity, optionally only from one chapter."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks (0 if the index does not exist)."""
