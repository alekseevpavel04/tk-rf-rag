from pathlib import Path

from app.embeddings import Embedder
from app.ingest.chunker import chunk_articles
from app.ingest.parser import load_articles
from app.store.base import VectorStore


def build_index(
    raw_text_path: str | Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    overlap: int,
) -> tuple[int, int]:
    """Parse -> chunk -> embed -> (re)create index. Returns (articles, chunks)."""
    path = Path(raw_text_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/download_tk.py first (see README)")
    articles = load_articles(path)
    chunks = chunk_articles(articles, chunk_size=chunk_size, overlap=overlap)
    vectors = embedder.embed_passages([c.text for c in chunks])
    store.recreate(embedder.dim)
    store.upsert(chunks, vectors)
    return len(articles), len(chunks)
