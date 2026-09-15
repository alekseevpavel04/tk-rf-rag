import pytest

from app.ingest.chunker import chunk_article, chunk_articles, split_text
from app.ingest.parser import Article


def make_article(text: str, number: str = "81") -> Article:
    return Article(number=number, title="Заголовок", chapter="13", chapter_title="Глава", text=text)


PARAGRAPHS = [f"Абзац номер {i}: " + "слово " * 20 for i in range(12)]
TEXT = "\n".join(p.strip() for p in PARAGRAPHS)


def test_short_text_is_single_chunk():
    assert split_text("Короткий текст.", chunk_size=500, overlap=50) == ["Короткий текст."]


@pytest.mark.parametrize("chunk_size,overlap", [(200, 0), (300, 60), (500, 150)])
def test_chunks_respect_size(chunk_size, overlap):
    pieces = split_text(TEXT, chunk_size, overlap)
    assert len(pieces) > 1
    assert all(len(p) <= chunk_size for p in pieces)


def test_all_paragraphs_are_covered():
    pieces = split_text(TEXT, chunk_size=300, overlap=60)
    joined = "\n".join(pieces)
    for p in PARAGRAPHS:
        assert p.strip() in joined


def test_overlap_repeats_tail_of_previous_chunk():
    pieces = split_text(TEXT, chunk_size=300, overlap=150)
    for prev, nxt in zip(pieces, pieces[1:]):
        last_paragraph = prev.split("\n")[-1]
        assert nxt.startswith(last_paragraph)


def test_zero_overlap_has_no_repeats():
    pieces = split_text(TEXT, chunk_size=300, overlap=0)
    units = [u for p in pieces for u in p.split("\n")]
    assert len(units) == len(set(units))


def test_long_paragraph_without_newlines_is_split():
    long_text = " ".join(["Очень длинное предложение без переносов строк."] * 40)
    pieces = split_text(long_text, chunk_size=200, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_params_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text(TEXT, chunk_size, overlap)


def test_chunk_metadata_and_header():
    chunks = chunk_article(make_article(TEXT), chunk_size=300, overlap=60)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.article == "81" and c.chapter == "13" for c in chunks)
    assert all(c.text.startswith("Статья 81. Заголовок\n") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunks_do_not_cross_articles():
    chunks = chunk_articles([make_article(TEXT, "80"), make_article(TEXT, "81")], chunk_size=300, overlap=60)
    for c in chunks:
        assert c.text.startswith(f"Статья {c.article}.")
    assert {c.article for c in chunks} == {"80", "81"}
