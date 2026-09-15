"""Sanity check for eval/questions.jsonl against the corpus.

For every question prints the gold article title and whether the key facts
(`must_contain` substrings passed below) actually occur in the article text.
Run after re-downloading the code: the law changes and gold answers may go stale.

Usage: python eval/check_questions.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.ingest.parser import load_articles  # noqa: E402

# key phrases that must be literally present in the gold article (lower-case, "ё" -> "е")
KEY_FACTS = {
    1: ["28 календарных дней"],
    2: ["две недели"],
    3: ["40 часов в неделю"],
    4: ["трех месяцев", "шести месяцев"],
    5: ["шестнадцати лет"],
    6: ["4 часов", "120 часов в год"],
    7: ["двойном размере"],
    8: ["прогула", "более четырех часов подряд"],
    9: ["замечание", "выговор", "увольнение"],
    10: ["одного месяца со дня обнаружения", "шести месяцев со дня"],
    11: ["каждые полмесяца"],
    12: ["одной сто пятидесятой"],
    13: ["не менее чем за два месяца"],
    14: ["среднего месячного заработка"],
    15: ["70", "86", "110"],
    16: ["трех лет"],
    17: ["соглашение между работодателем и работником"],
    18: ["место работы", "трудовая функция", "условия оплаты труда"],
    19: ["трех рабочих дней"],
    20: ["35 часов в неделю"],
    21: ["вне места нахождения работодателя"],
    22: ["20 процентов", "50 процентов", "70 процентов"],
    23: ["в день увольнения"],
    24: ["не более двух часов", "не менее 30 минут"],
    25: ["42 часов"],
}


def norm(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def main() -> None:
    articles = {a.number: a for a in load_articles(ROOT / get_settings().raw_text_path)}
    questions = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    problems = 0
    for q in questions:
        if not q["articles"]:
            print(f"#{q['id']:>2} [no answer] {q['question']}")
            continue
        text = " ".join(norm(articles[n].text) for n in q["articles"] if n in articles)
        missing_articles = [n for n in q["articles"] if n not in articles]
        missing_facts = [f for f in KEY_FACTS.get(q["id"], []) if norm(f) not in text]
        status = "OK " if not (missing_articles or missing_facts) else "BAD"
        problems += status == "BAD"
        titles = "; ".join(f"ст. {n} {articles[n].title}" for n in q["articles"] if n in articles)
        print(f"#{q['id']:>2} {status} {titles} | missing articles={missing_articles} facts={missing_facts}")
    print(f"\nproblems: {problems}")


if __name__ == "__main__":
    main()
