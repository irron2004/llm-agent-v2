from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.document_service import (  # noqa: E402
    DocumentIndexService,
    SourceDocument,
)
from backend.services.rag_service import RAGService  # noqa: E402
from backend.llm_infrastructure.retrieval.base import RetrievalResult  # noqa: E402


class _FakeEmbedder:
    def __init__(self, dim=2):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        return np.array([text.count("apple"), text.count("banana")], dtype=np.float32)

    def embed_batch(self, texts):
        return np.vstack([self.embed(t) for t in texts])


class _FakePreprocessor:
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def preprocess(self, texts):
        return [f"{t.lower().strip()}-{self.suffix}" for t in texts]


class _FakeChatService:
    def chat(self, message: str, *, system_prompt=None, **kwargs):
        # Return a fixed answer for predictability
        return type("Resp", (), {"text": f"answer:{message}", "raw": {}})


def _build_and_persist_corpus(tmp_path: Path, preproc_suffix: str):
    docs = [
        SourceDocument(doc_id="d1", text="Apple banana fruit"),
        SourceDocument(doc_id="d2", text="Banana smoothie recipe"),
    ]
    embedder = _FakeEmbedder()
    preproc = _FakePreprocessor(preproc_suffix)
    indexer = DocumentIndexService(
        embedder=embedder,
        preprocessor=preproc,
        normalize_vectors=True,
    )
    persist_dir = tmp_path / "corpus"
    corpus = indexer.index(
        docs,
        preprocess=True,
        build_sparse=True,
        persist_dir=persist_dir,
    )
    return corpus, persist_dir, embedder, preproc


def test_rag_service_reload_requires_same_embedder(tmp_path, monkeypatch):
    """Ensure reloaded corpus uses the same embedder to avoid dim mismatch."""
    corpus, persist_dir, embedder, preproc = _build_and_persist_corpus(tmp_path, "p1")

    # Simulate reload
    reloaded = DocumentIndexService.load(persist_dir)
    # Force SearchService to reuse the original embedder
    reloaded.embedder = embedder

    # Patch RAGService to bypass real LLM
    monkeypatch.setattr(
        "backend.services.rag_service.ChatService",
        lambda *_, **__: _FakeChatService(),
    )
    # Patch preprocessor to ensure the same one is used (important for matching)
    monkeypatch.setattr(
        "backend.services.rag_service.get_preprocessor",
        lambda *_, **__: preproc,
    )

    rag = RAGService(reloaded, retrieval_top_k=2)
    response = rag.query("apple?", top_k=2)

    assert len(response.context) == 2
    assert all(isinstance(r, RetrievalResult) for r in response.context)


def test_rag_service_reload_fails_with_different_preprocessor(tmp_path, monkeypatch):
    """Demonstrate mismatch when a different preprocessor is used after reload."""
    corpus, persist_dir, embedder, _ = _build_and_persist_corpus(tmp_path, "p1")
    reloaded = DocumentIndexService.load(persist_dir)
    reloaded.embedder = embedder

    bad_preproc = _FakePreprocessor("DIFF")

    monkeypatch.setattr(
        "backend.services.rag_service.ChatService",
        lambda *_, **__: _FakeChatService(),
    )
    # Inject a different preprocessor to show mismatch in metadata
    monkeypatch.setattr(
        "backend.services.rag_service.get_preprocessor",
        lambda *_, **__: bad_preproc,
    )

    rag = RAGService(reloaded, retrieval_top_k=1)
    resp = rag.query("apple?", top_k=1)
    # The preprocessed query should carry the injected suffix
    assert resp.metadata["preprocessed_query"].endswith("diff")
