Сравнение прогонов: `faiss_500_nofilter__questions_tk_hard__base.jsonl` (base) и `faiss_500_nofilter__e-projects-ru-law-retrieval-data-export-hf-model_39c3a7__questions_tk_hard__v2.jsonl` (ft-v2), вопросы `questions_tk_hard.jsonl`.

| Группа | n | Hit@1 base | Hit@1 ft-v2 | Hit@5 base | Hit@5 ft-v2 | MRR@5 base | MRR@5 ft-v2 | Δ MRR@5 [95% ДИ] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| все | 716 | 0.591 | 0.721 | 0.830 | 0.906 | 0.688 | 0.793 | +0.105 [+0.084; +0.127] |
| qtype=search | 280 | 0.754 | 0.857 | 0.954 | 0.979 | 0.838 | 0.909 | +0.071 [+0.042; +0.101] |
| slice=seen | 644 | 0.578 | 0.716 | 0.826 | 0.902 | 0.679 | 0.789 | +0.110 [+0.088; +0.133] |
| qtype=everyday | 436 | 0.486 | 0.633 | 0.750 | 0.860 | 0.591 | 0.719 | +0.128 [+0.098; +0.158] |
| slice=unseen_articles | 72 | 0.708 | 0.764 | 0.861 | 0.944 | 0.770 | 0.833 | +0.063 [-0.004; +0.134] |
