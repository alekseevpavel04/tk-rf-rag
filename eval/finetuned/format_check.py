"""Does the chunk format explain the difference between base and fine-tuned e5-small on TK?

The fine-tuned model (ru-law-retrieval) was trained on 600/90 chunks whose header starts with the code name
("Трудовой кодекс РФ, Статья N. ..."); tk-rf-rag uses 500/75 chunks with "Статья N. ..." headers.
Grid: model x chunk size x code-name prefix x question set. Retrieval as in run_eval.py (20 chunks ->
unique articles -> Hit@1 / MRR@5 over top-5 articles), no chapter filter.

Usage: python eval/finetuned/format_check.py  (writes eval/finetuned/format_check.md)
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from app.embeddings import Embedder
from app.ingest.parser import load_articles
from app.ingest.chunker import chunk_articles

arts = load_articles(ROOT / "data/raw/tk_rf.txt")
qsets = {n: [json.loads(l) for l in open(ROOT / f"eval/{n}.jsonl", encoding="utf-8") if l.strip()] for n in ("questions", "questions_ru_law_tk")}
res = {}
for model in ("intfloat/multilingual-e5-small", "alekseevpavel04/multilingual-e5-small-ru-law"):
    emb = Embedder(model)
    for size, overlap in ((500, 75), (600, 90)):
        chunks = chunk_articles(arts, chunk_size=size, overlap=overlap)
        for prefix in ("", "Трудовой кодекс РФ, "):
            P = emb.embed_passages([prefix + c.text for c in chunks])
            for qn, qs in qsets.items():
                qs = [q for q in qs if q["articles"]]
                Q = np.stack([emb.embed_query(q["question"]) for q in qs])
                S = Q @ P.T
                h1 = rr = 0.0
                for i, q in enumerate(qs):
                    top = np.argsort(-S[i])[:20]
                    ranked = list(dict.fromkeys(chunks[j].article for j in top))[:5]
                    gold = set(q["articles"])
                    rank = next((k for k, a in enumerate(ranked, 1) if a in gold), None)
                    h1 += ranked[0] in gold; rr += 1 / rank if rank else 0
                key = f"{model.split('/')[-1]:32} {size}/{overlap} prefix={'yes' if prefix else 'no ':3} {qn:22}"
                res[key] = (h1 / len(qs), rr / len(qs))
                print(key, f"Hit@1 {h1/len(qs):.3f}  MRR@5 {rr/len(qs):.3f}", flush=True)

lines = ["| Модель | Фрагменты | Префикс кодекса | Вопросы | Hit@1 | MRR@5 |", "|---|---|---|---|---:|---:|"]
for key, (h1, mrr) in res.items():
    model, chunking, prefix, qn = key.split()
    lines.append(f"| {model} | {chunking} | {prefix.split('=')[1]} | {qn} | {h1:.3f} | {mrr:.3f} |")
(ROOT / "eval/finetuned/format_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
