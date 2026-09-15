"""multilingual-e5 embeddings.

E5 models were trained with prefixes: "query: " for search queries and
"passage: " for documents. Without them retrieval quality drops noticeably.
Vectors are L2-normalized, so dot product == cosine similarity.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + text])[0]


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)
