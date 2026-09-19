import asyncio
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings, get_settings
from app.embeddings import get_embedder
from app.ingest.pipeline import build_index
from app.llm import OpenAICompatibleLLM
from app.rag import RAGService
from app.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse
from app.store import create_store

logger = logging.getLogger("tk_rag")
_ingest_lock = asyncio.Lock()


@lru_cache
def get_service() -> RAGService:
    settings = get_settings()
    return RAGService(
        embedder=get_embedder(settings.embedding_model),
        store=create_store(settings),
        llm=OpenAICompatibleLLM.from_settings(settings),
        min_score=settings.min_score,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: load the embedding model at startup, not on the first request.
    factory = app.dependency_overrides.get(get_service, get_service)
    await asyncio.to_thread(factory)
    yield


app = FastAPI(title="TK RF RAG", description="Вопросы-ответы по Трудовому кодексу РФ", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(service: RAGService = Depends(get_service)) -> HealthResponse:
    model = getattr(service.embedder, "model_name", "")
    try:
        count = await asyncio.to_thread(service.store.count)
    except Exception as exc:  # vector DB is down
        logger.warning("vector store unavailable: %s", exc)
        return HealthResponse(
            status="vector_store_unavailable", vector_store=service.store.name, indexed_chunks=0, embedding_model=model
        )
    return HealthResponse(status="ok", vector_store=service.store.name, indexed_chunks=count, embedding_model=model)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    service: RAGService = Depends(get_service), settings: Settings = Depends(get_settings)
) -> IngestResponse:
    if _ingest_lock.locked():
        raise HTTPException(409, "Indexing is already running")
    async with _ingest_lock:
        started = time.perf_counter()
        try:
            articles, chunks = await asyncio.to_thread(
                build_index,
                settings.raw_text_path,
                service.embedder,
                service.store,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
    return IngestResponse(
        articles=articles,
        chunks=chunks,
        vector_store=service.store.name,
        seconds=round(time.perf_counter() - started, 2),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, service: RAGService = Depends(get_service)) -> AskResponse:
    if await asyncio.to_thread(service.store.count) == 0:
        raise HTTPException(409, "Index is empty: call POST /ingest first")
    try:
        return await service.ask(request.question, top_k=request.top_k, chapter=request.chapter)
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(502, f"LLM or vector store error: {type(exc).__name__}") from exc
