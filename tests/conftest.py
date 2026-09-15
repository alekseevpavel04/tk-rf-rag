import hashlib
import re

import numpy as np
import pytest

from app.ingest.chunker import chunk_articles
from app.ingest.parser import parse_articles
from app.rag import RAGService
from app.store.faiss_store import FaissStore

SAMPLE_TK = """Глава 13. Прекращение трудового договора

Статья 80. Расторжение трудового договора по инициативе работника (по собственному желанию)
Работник имеет право расторгнуть трудовой договор, предупредив об этом работодателя в письменной форме не позднее чем за две недели.

Статья 81. Расторжение трудового договора по инициативе работодателя
Трудовой договор может быть расторгнут работодателем в случае прогула.

Глава 19. Отпуска

Статья 115. Продолжительность ежегодного основного оплачиваемого отпуска
Ежегодный основной оплачиваемый отпуск предоставляется работникам продолжительностью 28 календарных дней.
"""


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download in tests."""

    dim = 128

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w{4,}", text.lower()):
            v[int(hashlib.md5(token[:6].encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    def __init__(self, answer: str = "Отпуск составляет 28 календарных дней (ст. 115 ТК РФ)."):
        self.answer = answer
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        self.calls.append(messages)
        return self.answer


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def indexed_store(embedder):
    chunks = chunk_articles(parse_articles(SAMPLE_TK), chunk_size=500, overlap=50)
    store = FaissStore(directory=None)
    store.recreate(embedder.dim)
    store.upsert(chunks, embedder.embed_passages([c.text for c in chunks]))
    return store


@pytest.fixture
def service(embedder, indexed_store, llm):
    return RAGService(embedder=embedder, store=indexed_store, llm=llm)
