import numpy as np
import pytest

from app.ingest.chunker import Chunk
from app.store.faiss_store import FaissStore
from app.store.qdrant_store import QdrantStore

DIM = 16


def make_data(n: int = 20):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = [
        Chunk(
            id=f"00000000-0000-0000-0000-{i:012d}",
            article=str(i),
            title=f"Статья {i}",
            chapter="1" if i < n // 2 else "2",
            chapter_title="",
            position=0,
            text=f"text {i}",
        )
        for i in range(n)
    ]
    return chunks, vectors


@pytest.fixture(params=["qdrant", "faiss"])
def store(request, tmp_path):
    if request.param == "qdrant":
        return QdrantStore(url=":memory:", collection="test")
    return FaissStore(directory=str(tmp_path / "faiss"))


def test_empty_store_count_is_zero(store):
    assert store.count() == 0


def test_upsert_and_nearest_is_itself(store):
    chunks, vectors = make_data()
    store.recreate(DIM)
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)
    hits = store.search(vectors[7], top_k=3)
    assert len(hits) == 3
    assert hits[0].payload["article"] == "7"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_chapter_filter(store):
    chunks, vectors = make_data()
    store.recreate(DIM)
    store.upsert(chunks, vectors)
    hits = store.search(vectors[3], top_k=5, chapter="2")
    assert hits and all(h.payload["chapter"] == "2" for h in hits)
    assert store.search(vectors[3], top_k=5, chapter="999") == []


def test_recreate_drops_old_data(store):
    chunks, vectors = make_data()
    store.recreate(DIM)
    store.upsert(chunks, vectors)
    store.recreate(DIM)
    assert store.count() == 0


def test_qdrant_and_faiss_return_same_ranking(tmp_path):
    chunks, vectors = make_data(50)
    q, f = QdrantStore(url=":memory:", collection="t"), FaissStore(directory=None)
    for s in (q, f):
        s.recreate(DIM)
        s.upsert(chunks, vectors)
    query = vectors[0] + 0.3 * vectors[1]
    query /= np.linalg.norm(query)
    q_hits, f_hits = q.search(query, top_k=10), f.search(query, top_k=10)
    assert [h.payload["article"] for h in q_hits] == [h.payload["article"] for h in f_hits]
    assert [h.score for h in q_hits] == pytest.approx([h.score for h in f_hits], abs=1e-4)


def test_faiss_persists_to_disk(tmp_path):
    chunks, vectors = make_data()
    s = FaissStore(directory=str(tmp_path / "idx"))
    s.recreate(DIM)
    s.upsert(chunks, vectors)
    reloaded = FaissStore(directory=str(tmp_path / "idx"))
    assert reloaded.count() == len(chunks)
    assert reloaded.search(vectors[5], top_k=1)[0].payload["article"] == "5"
