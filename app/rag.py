import asyncio
import re

from app.embeddings import Embedder
from app.llm import LLM
from app.prompts import REFUSAL, build_answer_messages
from app.schemas import AskResponse, Source
from app.store.base import SearchHit, VectorStore

SNIPPET_LEN = 300

_NUM = r"\d+(?:[.\-]\d+)*"  # 81, 22.1, 341.1-1
ARTICLE_REF_RE = re.compile(
    rf"(?:\bст\.|\bстать(?:я|и|е|ю|ей|ям|ях|ями))\s*({_NUM}(?:\s*(?:,|и)\s*{_NUM})*)",
    re.IGNORECASE,
)


def extract_article_refs(text: str) -> set[str]:
    """'(ст. 81, 82 ТК РФ) ... статьи 22.1' -> {'81', '82', '22.1'}"""
    refs: set[str] = set()
    for group in ARTICLE_REF_RE.findall(text):
        refs.update(re.findall(_NUM, group))
    return refs


def is_refusal(text: str) -> bool:
    return "не нашел" in text.lower().replace("ё", "е")


def unique_articles(hits: list[SearchHit]) -> list[str]:
    """Article numbers in rank order without duplicates (several chunks may share one article)."""
    return list(dict.fromkeys(h.payload["article"] for h in hits))


def to_sources(hits: list[SearchHit]) -> list[Source]:
    sources = []
    for h in hits:
        body = h.payload["text"].split("\n", 1)[-1]  # drop "Статья N. Title" header
        snippet = body if len(body) <= SNIPPET_LEN else body[:SNIPPET_LEN].rstrip() + "…"
        sources.append(
            Source(article=h.payload["article"], title=h.payload["title"], score=round(h.score, 4), text_snippet=snippet)
        )
    return sources


class RAGService:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLM, min_score: float = 0.0):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.min_score = min_score

    def retrieve(self, question: str, top_k: int, chapter: str | None = None) -> list[SearchHit]:
        vector = self.embedder.embed_query(question)
        hits = self.store.search(vector, top_k=top_k, chapter=chapter)
        return [h for h in hits if h.score >= self.min_score]

    async def generate(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return REFUSAL  # nothing to ground the answer on: do not even call the LLM
        answer = await self.llm.complete(build_answer_messages(question, hits))
        return answer or REFUSAL

    async def ask(self, question: str, top_k: int, chapter: str | None = None) -> AskResponse:
        hits = await asyncio.to_thread(self.retrieve, question, top_k, chapter)
        answer = await self.generate(question, hits)
        return AskResponse(answer=answer, sources=to_sources(hits))
