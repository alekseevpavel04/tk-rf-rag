"""FAISS implementation: exact inner-product search over normalized vectors (= cosine).

FAISS is a library, not a database: it stores only vectors. Payloads live next
to the index in a JSON file, and the chapter filter is done with an IDSelector
(search only among ids that belong to the chapter).
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.ingest.chunker import Chunk
from app.store.base import SearchHit, VectorStore


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, directory: str | None):
        # directory=None keeps everything in memory (used in tests)
        self.directory = Path(directory) if directory else None
        self.index: faiss.Index | None = None
        self.payloads: list[dict] = []
        self._load()

    @property
    def _index_path(self) -> Path:
        return self.directory / "index.faiss"

    @property
    def _payload_path(self) -> Path:
        return self.directory / "payloads.json"

    def _load(self) -> None:
        if self.directory and self._index_path.exists() and self._payload_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            self.payloads = json.loads(self._payload_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if not self.directory:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))
        self._payload_path.write_text(json.dumps(self.payloads, ensure_ascii=False), encoding="utf-8")

    def recreate(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.payloads = []
        self._save()

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if self.index is None:
            self.recreate(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.payloads.extend(c.payload() for c in chunks)
        self._save()

    def search(self, vector: np.ndarray, top_k: int, chapter: str | None = None) -> list[SearchHit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        params = None
        if chapter:
            ids = np.array([i for i, p in enumerate(self.payloads) if p.get("chapter") == chapter], dtype=np.int64)
            if ids.size == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
        scores, idxs = self.index.search(query, top_k, params=params)
        return [
            SearchHit(score=float(s), payload=self.payloads[i])
            for s, i in zip(scores[0], idxs[0])
            if i != -1
        ]

    def count(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)
