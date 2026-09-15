"""(Re)build the vector index from data/raw/tk_rf.txt.

Usage:
    python scripts/ingest.py                         # settings from .env
    python scripts/ingest.py --store faiss --chunk-size 500 --overlap 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.ingest.pipeline import build_index  # noqa: E402
from app.store import create_store  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=["qdrant", "faiss"], default=settings.vector_store)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    settings = settings.model_copy(update={"vector_store": args.store})
    store = create_store(settings, name=args.collection)
    started = time.perf_counter()
    articles, chunks = build_index(
        settings.raw_text_path, get_embedder(settings.embedding_model), store, args.chunk_size, args.overlap
    )
    print(
        f"{store.name}/{args.collection}: {articles} articles -> {chunks} chunks "
        f"(size={args.chunk_size}, overlap={args.overlap}) in {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
