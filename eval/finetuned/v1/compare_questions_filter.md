Сравнение прогонов: `faiss_500_filter.jsonl` (base) и `faiss_500_filter__alekseevpavel04-multilingual-e5-small-ru-law_a095fa.jsonl` (ft), вопросы `questions.jsonl`.

| Группа | n | Hit@1 base | Hit@1 ft | Hit@5 base | Hit@5 ft | MRR@5 base | MRR@5 ft | Δ MRR@5 [95% ДИ] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| все | 25 | 1.000 | 0.920 | 1.000 | 1.000 | 1.000 | 0.953 | -0.047 [-0.120; +0.000] |
