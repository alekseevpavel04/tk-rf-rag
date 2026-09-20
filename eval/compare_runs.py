"""Paired comparison of two retrieval runs (per-question logs in eval/runs/*.jsonl).

Hit@1 / Hit@5 / MRR@5 for both runs, by question type and slice when the question file has them,
and a paired bootstrap 95% CI of the MRR@5 difference (10 000 resamples over questions).

Usage:
    python eval/compare_runs.py --a eval/runs/A.jsonl --b eval/runs/B.jsonl \
        --questions eval/questions_ru_law_tk.jsonl --label-a base --label-b finetuned --out eval/finetuned/x.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows if "hit1" in r}


def bootstrap_ci(diff: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    a, b = load(args.a), load(args.b)
    questions = {q["id"]: q for q in map(json.loads, Path(args.questions).read_text(encoding="utf-8").splitlines())}
    ids = sorted(set(a) & set(b))
    groups: dict[str, list[int]] = defaultdict(list)
    for i in ids:
        groups["все"].append(i)
        for key in ("qtype", "slice"):
            if key in questions[i]:
                groups[f"{key}={questions[i][key]}"].append(i)

    lines = [
        f"Сравнение прогонов: `{Path(args.a).name}` ({args.label_a}) и `{Path(args.b).name}` ({args.label_b}), "
        f"вопросы `{Path(args.questions).name}`.",
        "",
        f"| Группа | n | Hit@1 {args.label_a} | Hit@1 {args.label_b} | Hit@5 {args.label_a} | Hit@5 {args.label_b} "
        f"| MRR@5 {args.label_a} | MRR@5 {args.label_b} | Δ MRR@5 [95% ДИ] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g in groups.items():
        ra, rb = np.array([a[i]["rr"] for i in g]), np.array([b[i]["rr"] for i in g])
        lo, hi = bootstrap_ci(rb - ra)
        lines.append(
            f"| {name} | {len(g)} | {np.mean([a[i]['hit1'] for i in g]):.3f} | {np.mean([b[i]['hit1'] for i in g]):.3f} "
            f"| {np.mean([a[i]['hit5'] for i in g]):.3f} | {np.mean([b[i]['hit5'] for i in g]):.3f} "
            f"| {ra.mean():.3f} | {rb.mean():.3f} | {rb.mean() - ra.mean():+.3f} [{lo:+.3f}; {hi:+.3f}] |"
        )
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
