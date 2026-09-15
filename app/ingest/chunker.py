"""Split articles into overlapping chunks.

Chunks never cross article boundaries, so every chunk has exactly one article
number. Text is packed greedily from paragraphs (then sentences, then words for
very long pieces) up to `chunk_size` characters; the next chunk starts with the
tail of the previous one, up to `overlap` characters.
"""

import re
import uuid
from dataclasses import dataclass

from app.ingest.parser import Article

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+")


@dataclass
class Chunk:
    id: str
    article: str
    title: str
    chapter: str
    chapter_title: str
    position: int  # index of chunk inside the article
    text: str  # "Статья N. Title\n" + body piece; this is what gets embedded

    def payload(self) -> dict:
        return {
            "article": self.article,
            "title": self.title,
            "chapter": self.chapter,
            "chapter_title": self.chapter_title,
            "position": self.position,
            "text": self.text,
        }


def _split_to_units(text: str, chunk_size: int) -> list[str]:
    """Paragraphs -> sentences -> words, so that every unit fits into chunk_size."""
    units: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in SENTENCE_SPLIT_RE.split(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue
            piece = ""
            for word in sentence.split():
                if piece and len(piece) + 1 + len(word) > chunk_size:
                    units.append(piece)
                    piece = word
                else:
                    piece = f"{piece} {word}" if piece else word
            if piece:
                units.append(piece)
    return units


def _joined_len(units: list[str]) -> int:
    return sum(len(u) for u in units) + max(len(units) - 1, 0)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    pieces: list[str] = []
    current: list[str] = []
    has_new = False  # current contains something not yet emitted
    for unit in _split_to_units(text, chunk_size):
        if current and _joined_len(current + [unit]) > chunk_size:
            pieces.append("\n".join(current))
            tail: list[str] = []
            for prev in reversed(current):
                if _joined_len([prev] + tail) > overlap:
                    break
                tail.insert(0, prev)
            current = tail if _joined_len(tail + [unit]) <= chunk_size else []
        current.append(unit)
        has_new = True
    if current and has_new:
        pieces.append("\n".join(current))
    return pieces


def chunk_article(article: Article, chunk_size: int, overlap: int) -> list[Chunk]:
    body = article.text or article.title
    return [
        Chunk(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"tk-rf/{article.number}/{chunk_size}/{overlap}/{i}")),
            article=article.number,
            title=article.title,
            chapter=article.chapter,
            chapter_title=article.chapter_title,
            position=i,
            text=f"{article.header}\n{piece}",
        )
        for i, piece in enumerate(split_text(body, chunk_size, overlap))
    ]


def chunk_articles(articles: list[Article], chunk_size: int, overlap: int) -> list[Chunk]:
    return [chunk for article in articles for chunk in chunk_article(article, chunk_size, overlap)]
