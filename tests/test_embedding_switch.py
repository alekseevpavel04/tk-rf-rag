"""Switching the embedding model: config, per-model index names, prefixes, /health."""

import os
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.embeddings as embeddings
from app.config import DEFAULT_EMBEDDING_MODEL, FINETUNED_EMBEDDING_MODEL, Settings
from app.main import app, get_service
from app.store import create_store, index_name


def test_default_model_is_base_e5_small(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert Settings(_env_file=None).embedding_model == DEFAULT_EMBEDDING_MODEL


def test_model_switched_by_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", FINETUNED_EMBEDDING_MODEL)
    assert Settings(_env_file=None).embedding_model == FINETUNED_EMBEDDING_MODEL


def test_base_model_keeps_old_index_name():
    assert index_name(Settings(_env_file=None, embedding_model=DEFAULT_EMBEDDING_MODEL)) == "tk_rf"


def test_other_models_get_their_own_index():
    ft = index_name(Settings(_env_file=None, embedding_model=FINETUNED_EMBEDDING_MODEL))
    local_a = index_name(Settings(_env_file=None, embedding_model=r"D:\models\run_a\best"))
    local_b = index_name(Settings(_env_file=None, embedding_model=r"D:\models\run_b\best"))
    assert ft.startswith("tk_rf__") and ft != "tk_rf"
    assert len({ft, local_a, local_b}) == 3
    for name in (ft, local_a, local_b):  # valid Qdrant collection / directory name
        assert re.fullmatch(r"[a-z0-9_\-]+", name.lower()) and len(name) < 80
    assert ft == index_name(Settings(_env_file=None, embedding_model=FINETUNED_EMBEDDING_MODEL))  # deterministic


def test_faiss_store_directory_follows_model(tmp_path):
    base = Settings(_env_file=None, vector_store="faiss", faiss_dir=str(tmp_path))
    ft = base.model_copy(update={"embedding_model": FINETUNED_EMBEDDING_MODEL})
    assert create_store(base).directory != create_store(ft).directory
    assert create_store(ft).directory.name == index_name(ft)


class _FakeST:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.seen: list[str] = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return np.ones((len(texts), 4), dtype=np.float32) / 2


@pytest.mark.parametrize("model", [DEFAULT_EMBEDDING_MODEL, FINETUNED_EMBEDDING_MODEL])
def test_e5_prefixes_are_kept_for_both_models(monkeypatch, model):
    monkeypatch.setattr(embeddings, "SentenceTransformer", _FakeST)
    emb = embeddings.Embedder(model)
    emb.embed_query("вопрос")
    emb.embed_passages(["статья"])
    assert emb.model_name == model
    assert emb.model.seen == ["query: вопрос", "passage: статья"]


def test_health_reports_embedding_model(service):
    service.embedder.model_name = FINETUNED_EMBEDDING_MODEL
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as c:
        body = c.get("/health").json()
    app.dependency_overrides.clear()
    assert body["embedding_model"] == FINETUNED_EMBEDDING_MODEL


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="downloads the model; set RUN_MODEL_TESTS=1")
def test_finetuned_model_ranks_the_right_article():
    emb = embeddings.Embedder(FINETUNED_EMBEDDING_MODEL)
    q = emb.embed_query("Сколько дней длится ежегодный оплачиваемый отпуск?")
    p = emb.embed_passages(
        [
            "Статья 115. Продолжительность ежегодного основного оплачиваемого отпуска\n"
            "Ежегодный основной оплачиваемый отпуск предоставляется работникам продолжительностью 28 календарных дней.",
            "Статья 81. Расторжение трудового договора по инициативе работодателя\n"
            "Трудовой договор может быть расторгнут работодателем в случае прогула.",
        ]
    )
    assert emb.dim == 384
    scores = p @ q
    assert scores[0] > scores[1]
