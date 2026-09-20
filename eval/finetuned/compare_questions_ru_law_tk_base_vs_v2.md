Сравнение прогонов: `faiss_500_nofilter__questions_ru_law_tk__base.jsonl` (base) и `faiss_500_nofilter__e-projects-ru-law-retrieval-data-export-hf-model_39c3a7__questions_ru_law_tk__v2.jsonl` (ft-v2), вопросы `questions_ru_law_tk.jsonl`.

| Группа | n | Hit@1 base | Hit@1 ft-v2 | Hit@5 base | Hit@5 ft-v2 | MRR@5 base | MRR@5 ft-v2 | Δ MRR@5 [95% ДИ] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| все | 74 | 0.770 | 0.797 | 0.973 | 0.946 | 0.857 | 0.859 | +0.002 [-0.046; +0.048] |
| qtype=legal | 32 | 0.781 | 0.844 | 1.000 | 0.969 | 0.880 | 0.896 | +0.016 [-0.026; +0.068] |
| slice=seen | 43 | 0.767 | 0.814 | 0.953 | 0.953 | 0.843 | 0.861 | +0.018 [-0.010; +0.056] |
| qtype=search | 23 | 0.870 | 0.870 | 0.957 | 0.957 | 0.906 | 0.906 | +0.000 [+0.000; +0.000] |
| qtype=everyday | 19 | 0.632 | 0.632 | 0.947 | 0.895 | 0.759 | 0.739 | -0.020 [-0.193; +0.136] |
| slice=unseen_articles | 31 | 0.774 | 0.774 | 1.000 | 0.935 | 0.876 | 0.855 | -0.021 [-0.129; +0.075] |
