"""Tests for EmbeddingService."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.embedding_service import EmbeddingService  # noqa: E402


class _FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, 2.0], dtype=np.float32)

    def embed_batch(self, texts):
        self.calls += 1
        return np.array([[1.0, 2.0] for _ in texts], dtype=np.float32)

    def get_dimension(self) -> int:
        self.calls += 1
        return 2


def test_embedding_service_reuses_embedder(monkeypatch):
    fake = _FakeEmbedder()

    def _fake_get_embedder(method, version="v1", **kwargs):
        return fake

    # EmbeddingService가 import한 곳을 패치해야 함
    monkeypatch.setattr(
        "backend.services.embedding_service.get_embedder",
        _fake_get_embedder,
    )

    svc = EmbeddingService(method="bge_base", version="v1", device="cpu")

    vec = svc.embed_query("hello")
    vecs = svc.embed_texts(["a", "b"])
    dim = svc.dimension()

    assert vec.shape == (2,)
    assert vecs.shape == (2, 2)
    assert dim == 2
    # embedder는 한 번만 생성되어 호출 횟수는 3회(단일+배치+dimension)
    assert fake.calls == 3
