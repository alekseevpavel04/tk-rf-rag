import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_service
from app.rag import RAGService
from app.store.faiss_store import FaissStore
from tests.conftest import SAMPLE_TK


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_store"] == "faiss"
    assert body["indexed_chunks"] == 3


def test_ask_returns_answer_and_sources(client, llm):
    resp = client.post("/ask", json={"question": "Сколько дней ежегодный оплачиваемый отпуск?", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == llm.answer
    assert len(body["sources"]) == 2
    assert body["sources"][0]["article"] == "115"
    assert set(body["sources"][0]) == {"article", "title", "score", "text_snippet"}
    assert "Статья 115" not in body["sources"][0]["text_snippet"]  # header is stripped from snippet
    # the LLM received the retrieved context inside <context>
    user_msg = llm.calls[0][1]["content"]
    assert "<context>" in user_msg and 'article="115"' in user_msg


def test_ask_with_chapter_filter(client):
    resp = client.post("/ask", json={"question": "отпуск", "top_k": 5, "chapter": "13"})
    assert resp.status_code == 200
    assert {s["article"] for s in resp.json()["sources"]} <= {"80", "81"}


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "отпуск", "top_k": 0}, {"top_k": 3}])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_on_empty_index_returns_409(embedder, llm):
    app.dependency_overrides[get_service] = lambda: RAGService(embedder, FaissStore(directory=None), llm)
    with TestClient(app) as c:
        assert c.post("/ask", json={"question": "Сколько дней отпуск?"}).status_code == 409
    app.dependency_overrides.clear()
    assert llm.calls == []


def test_ingest_builds_index(tmp_path, embedder, llm):
    raw = tmp_path / "tk.txt"
    raw.write_text(SAMPLE_TK, encoding="utf-8")
    service = RAGService(embedder, FaissStore(directory=None), llm)
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: Settings(raw_text_path=str(raw), chunk_size=500, chunk_overlap=50)
    with TestClient(app) as c:
        resp = c.post("/ingest")
        assert resp.status_code == 200
        assert resp.json()["articles"] == 3
        assert resp.json()["chunks"] == 3
        assert c.get("/health").json()["indexed_chunks"] == 3
    app.dependency_overrides.clear()


def test_ingest_missing_file_returns_400(tmp_path, service):
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: Settings(raw_text_path=str(tmp_path / "nope.txt"))
    with TestClient(app) as c:
        assert c.post("/ingest").status_code == 400
    app.dependency_overrides.clear()
