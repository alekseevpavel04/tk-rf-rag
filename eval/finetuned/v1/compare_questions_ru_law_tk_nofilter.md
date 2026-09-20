Сравнение прогонов: `faiss_500_nofilter__questions_ru_law_tk.jsonl` (base) и `faiss_500_nofilter__alekseevpavel04-multilingual-e5-small-ru-law_a095fa__questions_ru_law_tk.jsonl` (ft), вопросы `questions_ru_law_tk.jsonl`.

| Группа | n | Hit@1 base | Hit@1 ft | Hit@5 base | Hit@5 ft | MRR@5 base | MRR@5 ft | Δ MRR@5 [95% ДИ] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| все | 74 | 0.770 | 0.716 | 0.973 | 0.946 | 0.857 | 0.814 | -0.043 [-0.096; +0.007] |
| qtype=legal | 32 | 0.781 | 0.781 | 1.000 | 1.000 | 0.880 | 0.867 | -0.013 [-0.057; +0.034] |
| slice=seen | 43 | 0.767 | 0.721 | 0.953 | 0.930 | 0.843 | 0.810 | -0.033 [-0.081; +0.014] |
| qtype=search | 23 | 0.870 | 0.783 | 0.957 | 0.957 | 0.906 | 0.862 | -0.043 [-0.109; +0.000] |
| qtype=everyday | 19 | 0.632 | 0.526 | 0.947 | 0.842 | 0.759 | 0.667 | -0.092 [-0.268; +0.070] |
| slice=unseen_articles | 31 | 0.774 | 0.710 | 1.000 | 0.968 | 0.876 | 0.820 | -0.056 [-0.164; +0.038] |
