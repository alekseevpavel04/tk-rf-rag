"""Check that chunks fit into the 512-token limit of multilingual-e5 (longer input is silently truncated).

Usage: python eval/token_lengths.py --chunk-sizes 500 1000
"""

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embeddings import PASSAGE_PREFIX  # noqa: E402
from app.ingest.chunker import chunk_articles  # noqa: E402
from app.ingest.parser import load_articles  # noqa: E402

MAX_TOKENS = 512


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    args = parser.parse_args()

    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    articles = load_articles(ROOT / settings.raw_text_path)
    print(f"articles: {len(articles)}")
    for size in args.chunk_sizes:
        chunks = chunk_articles(articles, size, int(size * args.overlap_ratio))
        lengths = [len(tokenizer(PASSAGE_PREFIX + c.text)["input_ids"]) for c in chunks]
        chars = [len(c.text) for c in chunks]
        over = sum(n > MAX_TOKENS for n in lengths)
        print(
            f"chunk_size={size}: chunks={len(chunks)}, chars median={statistics.median(chars):.0f} max={max(chars)}, "
            f"tokens median={statistics.median(lengths):.0f} p95={sorted(lengths)[int(0.95 * len(lengths))]} "
            f"max={max(lengths)}, over {MAX_TOKENS}: {over}"
        )


if __name__ == "__main__":
    main()
