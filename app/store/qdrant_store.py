import numpy as np
from qdrant_client import QdrantClient, models

from app.ingest.chunker import Chunk
from app.store.base import SearchHit, VectorStore


class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(self, url: str, collection: str, client: QdrantClient | None = None):
        # url=":memory:" runs an embedded in-process Qdrant (used in tests)
        if client is not None:
            self.client = client
        elif url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self.client.create_payload_index(self.collection, "chapter", models.PayloadSchemaType.KEYWORD)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch_size: int = 256) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=c.id, vector=v.tolist(), payload=c.payload())
                    for c, v in zip(batch, vectors[start : start + batch_size])
                ],
                wait=True,
            )

    def search(self, vector: np.ndarray, top_k: int, chapter: str | None = None) -> list[SearchHit]:
        query_filter = None
        if chapter:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="chapter", match=models.MatchValue(value=chapter))]
            )
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [SearchHit(score=float(p.score), payload=dict(p.payload or {})) for p in response.points]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count
