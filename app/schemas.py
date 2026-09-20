from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    chapter: str | None = Field(default=None, description="Optional filter by chapter number, e.g. '19'")


class Source(BaseModel):
    article: str
    title: str
    score: float
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    articles: int
    chunks: int
    vector_store: str
    seconds: float


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    indexed_chunks: int
    embedding_model: str = ""
