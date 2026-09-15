"""Sanity check of the LLM judge: does it ever say 0?

Takes real answers from an eval run log, builds a corrupted version of each
(every number in the answer is replaced with a different one, e.g. "28 дней" ->
"31 дней"), and asks the judge about both versions with the same fragments.

A useful judge says 1 for originals and 0 for corrupted answers. If it says 1
for corrupted answers, faithfulness in results.md is not informative.

Usage: python eval/judge_sanity.py --run eval/runs/qdrant_500_nofilter.jsonl --chunk-size 500
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402
from app.prompts import build_judge_messages  # noqa: E402
from app.rag import RAGService  # noqa: E402
from app.store import create_store  # noqa: E402

REF_RE = re.compile(r"\((?:ст\.|стать)[^)]*\)")


def corrupt(answer: str) -> str | None:
    """Change every number outside article references. None if there is nothing to change."""
    refs = REF_RE.findall(answer)
    body = REF_RE.sub("\0", answer)
    changed, n = re.subn(r"\d+", lambda m: str(int(m.group()) + 3), body)
    if n == 0:
        return None
    for ref in refs:
        changed = changed.replace("\0", ref, 1)
    return changed


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(ROOT / "eval" / "runs" / "qdrant_500_nofilter.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    settings = get_settings()
    llm = OpenAICompatibleLLM.from_settings(settings)
    store = create_store(settings, name=f"eval_{args.chunk_size}")
    service = RAGService(get_embedder(settings.embedding_model), store, llm)

    records = [json.loads(l) for l in Path(args.run).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for r in records:
        if not r.get("gold") or r.get("refused") or "answer" not in r:
            continue
        bad = corrupt(r["answer"])
        if bad is None:
            continue
        hits = service.retrieve(r["question"], top_k=5, chapter=r.get("chapter_filter"))
        verdicts = []
        for text in (r["answer"], bad):
            raw = await llm.complete(build_judge_messages(text, hits), max_tokens=5)
            verdicts.append(next((int(c) for c in raw if c in "01"), None))
        rows.append((r["id"], verdicts[0], verdicts[1], bad))
        print(f"#{r['id']:>2} original={verdicts[0]} corrupted={verdicts[1]} | {bad[:110]}")

    n = len(rows)
    orig_ok = sum(v == 1 for _, v, _, _ in rows)
    caught = sum(v == 0 for _, _, v, _ in rows)
    print(f"\nanswers with numbers: {n}; judge=1 on originals: {orig_ok}/{n}; judge=0 on corrupted: {caught}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
