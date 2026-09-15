"""Evaluate retrieval and generation quality on eval/questions.jsonl.

Retrieval (answerable questions only):
    retrieve RETRIEVE_CHUNKS chunks -> deduplicate to a ranked list of articles ->
    Hit@1, Hit@5, MRR@5 against the gold article numbers.
Generation (all questions, top_k=5 chunks as in the API):
    citation rate   - answerable questions whose answer cites at least one gold article
    faithfulness    - LLM-as-a-judge 0/1 on answerable, non-refused answers
    refusal rate    - unanswerable questions answered with a refusal
    false refusals  - answerable questions answered with a refusal
    latency         - mean wall time of RAGService.ask (retrieval + LLM), judge excluded

Experiment grid: chunk sizes x (no filter | chapter filter). The filter uses the
chapter of the gold article, i.e. it is an ORACLE filter (upper bound), and is
not applied to unanswerable questions (they have no gold chapter).

Usage:
    python eval/run_eval.py                         # full grid, Qdrant, with LLM
    python eval/run_eval.py --retrieval-only        # no LLM needed
    python eval/run_eval.py --store faiss --chunk-sizes 1000 --retrieval-only
"""

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.ingest.parser import load_articles  # noqa: E402
from app.ingest.pipeline import build_index  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402
from app.prompts import build_judge_messages  # noqa: E402
from app.rag import RAGService, extract_article_refs, is_refusal, unique_articles  # noqa: E402
from app.store import create_store  # noqa: E402

RETRIEVE_CHUNKS = 20
GEN_TOP_K = 5


@dataclass
class Metrics:
    config: str
    chunks: int
    hit1: list[int] = field(default_factory=list)
    hit5: list[int] = field(default_factory=list)
    rr: list[float] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    faithful: list[int] = field(default_factory=list)
    refused_unanswerable: list[int] = field(default_factory=list)
    refused_answerable: list[int] = field(default_factory=list)
    judge_unparsed: int = 0
    latencies: list[float] = field(default_factory=list)


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_metrics(ranked_articles: list[str], gold: set[str]) -> tuple[int, int, float]:
    top5 = ranked_articles[:5]
    rank = next((i for i, a in enumerate(top5, 1) if a in gold), None)
    return int(bool(top5) and top5[0] in gold), int(rank is not None), (1.0 / rank if rank else 0.0)


def parse_judge(text: str) -> int | None:
    for ch in text.strip():
        if ch in "01":
            return int(ch)
    return None


def fmt(values: list, as_count: bool = True) -> str:
    if not values:
        return "—"
    mean = sum(values) / len(values)
    return f"{mean:.3f} ({sum(values)}/{len(values)})" if as_count else f"{mean:.3f}"


async def evaluate(service: RAGService, questions, chapter_of, config, n_chunks, use_filter, with_llm, log):
    m = Metrics(config=config, chunks=n_chunks)
    if with_llm:  # warm-up so that the first question does not pay for lazy initialization
        await service.ask("Что такое трудовой договор?", top_k=GEN_TOP_K)

    for q in questions:
        gold = set(q["articles"])
        chapter = chapter_of.get(q["articles"][0]) if use_filter and gold else None
        record = {"id": q["id"], "question": q["question"], "gold": q["articles"], "chapter_filter": chapter}

        if gold:
            hits = service.retrieve(q["question"], top_k=RETRIEVE_CHUNKS, chapter=chapter)
            ranked = unique_articles(hits)
            h1, h5, rr = rank_metrics(ranked, gold)
            m.hit1.append(h1)
            m.hit5.append(h5)
            m.rr.append(rr)
            record.update(retrieved_articles=ranked[:5], hit1=h1, hit5=h5, rr=round(rr, 3))

        if with_llm:
            started = time.perf_counter()
            response = await service.ask(q["question"], top_k=GEN_TOP_K, chapter=chapter)
            m.latencies.append(time.perf_counter() - started)
            answer = response.answer
            refused = is_refusal(answer)
            record.update(answer=answer, refused=refused, latency=round(m.latencies[-1], 2))
            if not gold:
                m.refused_unanswerable.append(int(refused))
            else:
                m.refused_answerable.append(int(refused))
                cited = int(bool(extract_article_refs(answer) & gold))
                m.cited.append(cited)
                record["cited_gold"] = cited
                if not refused:
                    hits = service.retrieve(q["question"], top_k=GEN_TOP_K, chapter=chapter)
                    verdict_text = await service.llm.complete(build_judge_messages(answer, hits), max_tokens=5)
                    verdict = parse_judge(verdict_text)
                    if verdict is None:
                        m.judge_unparsed += 1
                        verdict = 0
                    m.faithful.append(verdict)
                    record.update(judge_raw=verdict_text, faithful=verdict)
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    return m


def results_table(rows: list[Metrics], with_llm: bool) -> str:
    header = "| Конфигурация | Фрагментов | Hit@1 | Hit@5 | MRR@5 |"
    sep = "|---|---:|---:|---:|---:|"
    if with_llm:
        header += " Ссылка на верную статью | Faithfulness (судья) | Отказ без ответа | Ложные отказы | Ср. время ответа, с |"
        sep += "---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        line = f"| {r.config} | {r.chunks} | {fmt(r.hit1)} | {fmt(r.hit5)} | {fmt(r.rr, as_count=False)} |"
        if with_llm:
            lat = f"{statistics.mean(r.latencies):.2f}" if r.latencies else "—"
            line += (
                f" {fmt(r.cited)} | {fmt(r.faithful)} | {fmt(r.refused_unanswerable)} |"
                f" {fmt(r.refused_answerable)} | {lat} |"
            )
        lines.append(line)
    return "\n".join(lines)


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--store", choices=["qdrant", "faiss"], default=settings.vector_store)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[500, 1000])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-filter-runs", action="store_true", help="skip chapter-filter runs")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results_latest.md"))
    args = parser.parse_args()

    with_llm = not args.retrieval_only
    questions = load_questions(Path(args.questions))
    chapter_of = {a.number: a.chapter for a in load_articles(ROOT / settings.raw_text_path)}
    missing = sorted({a for q in questions for a in q["articles"] if a not in chapter_of})
    if missing:
        raise SystemExit(f"gold articles not found in the corpus: {missing}")

    settings = settings.model_copy(update={"vector_store": args.store})
    embedder = get_embedder(settings.embedding_model)
    llm = OpenAICompatibleLLM.from_settings(settings) if with_llm else None
    runs_dir = ROOT / "eval" / "runs"
    runs_dir.mkdir(exist_ok=True)

    rows: list[Metrics] = []
    for size in args.chunk_sizes:
        overlap = int(size * args.overlap_ratio)
        store = create_store(settings, name=f"eval_{size}")
        started = time.perf_counter()
        _, n_chunks = build_index(ROOT / settings.raw_text_path, embedder, store, size, overlap)
        print(f"[{args.store}] chunk_size={size} overlap={overlap}: {n_chunks} chunks, {time.perf_counter() - started:.0f}s")
        service = RAGService(embedder, store, llm, min_score=settings.min_score)
        for use_filter in [False] if args.no_filter_runs else [False, True]:
            config = f"{args.store}, {size}/{overlap}, {'фильтр по главе' if use_filter else 'без фильтра'}"
            log_path = runs_dir / f"{args.store}_{size}_{'filter' if use_filter else 'nofilter'}.jsonl"
            with log_path.open("w", encoding="utf-8") as log:
                m = await evaluate(service, questions, chapter_of, config, n_chunks, use_filter, with_llm, log)
            rows.append(m)
            print(results_table([m], with_llm).splitlines()[-1], flush=True)

    n_answerable = sum(1 for q in questions if q["articles"])
    meta = [
        f"Дата прогона: {datetime.now():%Y-%m-%d %H:%M}",
        f"Вопросов: {len(questions)} (с ответом: {n_answerable}, без ответа: {len(questions) - n_answerable})",
        f"Эмбеддинги: {settings.embedding_model}; хранилище: {args.store}",
        (
            f"LLM (ответ и судья): {settings.llm_model}, temperature={settings.llm_temperature}, "
            f"system role: {'да' if settings.llm_system_role else 'нет (инструкции в user-сообщении)'}"
        )
        if with_llm
        else "LLM: не использовалась (--retrieval-only)",
        f"Retrieval: {RETRIEVE_CHUNKS} фрагментов → уникальные статьи → метрики по топ-5 статей; генерация: top_k={GEN_TOP_K}",
        f"ОС: {platform.system()} {platform.version()}, Python {platform.python_version()}",
    ]
    if with_llm:
        meta.append("Нераспознанных ответов судьи (засчитаны как 0): " + ", ".join(f"{r.config}: {r.judge_unparsed}" for r in rows))
    text = "\n".join(f"- {line}" for line in meta) + "\n\n" + results_table(rows, with_llm) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    asyncio.run(main())
