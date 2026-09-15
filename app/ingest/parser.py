"""Parse the plain-text Labor Code into articles with metadata."""

import re
from dataclasses import dataclass
from pathlib import Path

CHAPTER_RE = re.compile(r"^Глава\s+(\d+(?:\.\d+)?)\.\s*(.*)$")
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s*(.*)$")
REPEALED_RE = re.compile(r"^(Утратила силу|Исключена)", re.IGNORECASE)


@dataclass
class Article:
    number: str  # "81", "22.1"
    title: str
    chapter: str  # "13", "49.1" or "" if unknown
    chapter_title: str
    text: str  # paragraphs joined with "\n"

    @property
    def header(self) -> str:
        return f"Статья {self.number}. {self.title}"


def parse_articles(raw: str, skip_repealed: bool = True) -> list[Article]:
    articles: list[Article] = []
    chapter, chapter_title = "", ""
    current: Article | None = None
    body: list[str] = []

    def flush() -> None:
        if current is None:
            return
        current.text = "\n".join(body).strip()
        if skip_repealed and not current.text and REPEALED_RE.match(current.title):
            return
        articles.append(current)

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if m := CHAPTER_RE.match(line):
            flush()
            current, body = None, []
            chapter, chapter_title = m.group(1), m.group(2).strip()
        elif m := ARTICLE_RE.match(line):
            flush()
            current = Article(m.group(1), m.group(2).strip(), chapter, chapter_title, "")
            body = []
        elif current is not None:
            body.append(line)
    flush()
    return articles


def load_articles(path: str | Path, skip_repealed: bool = True) -> list[Article]:
    return parse_articles(Path(path).read_text(encoding="utf-8"), skip_repealed=skip_repealed)
