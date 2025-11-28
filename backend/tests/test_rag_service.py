"""Tests for RAGService end-to-end pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.rag_service import RAGService, RAGResponse  # noqa: E402
from backend.services.document_service import (  # noqa: E402
    DocumentIndexService,
    SourceDocument,
)
from backend.llm_infrastructure.llm.base import LLMResponse  # noqa: E402


class _FakeEmbedder:
    """Deterministic embedder for testing."""

    def embed(self, text: str) -> np.ndarray:
        return np.array([text.count("apple"), text.count("banana")], dtype=np.float32)

    def embed_batch(self, texts):
        return np.vstack([self.embed(t) for t in texts])


class _FakePreprocessor:
    """Simple preprocessor for testing."""

    def preprocess(self, texts):
        return [t.lower().strip() for t in texts]


class _FakeChatService:
    """Fake chat service that echoes context."""

    def __init__(self):
        self.last_system_prompt = None
        self.last_message = None

    def chat(self, message: str, *, system_prompt=None, **kwargs):
        self.last_system_prompt = system_prompt
        self.last_message = message
        # Echo back the question with marker
        answer = f"Answer to: {message}"
        if system_prompt and "Context" in system_prompt:
            answer += " (with context)"
        return LLMResponse(text=answer, raw={"message": message})


def _build_test_corpus(tmp_path: Path):
    """Build a small test corpus."""
    docs = [
        SourceDocument(doc_id="d1", text="Apple banana fruit"),
        SourceDocument(doc_id="d2", text="Banana smoothie recipe"),
        SourceDocument(doc_id="d3", text="Apple pie baking"),
    ]
    indexer = DocumentIndexService(
        embedder=_FakeEmbedder(),
        preprocessor=None,
        normalize_vectors=True,
    )
    return indexer.index(docs, preprocess=False, build_sparse=True)


def test_rag_service_full_pipeline(tmp_path, monkeypatch):
    """Test RAG service executes full pipeline."""
    corpus = _build_test_corpus(tmp_path)

    # Stub SearchService's EmbeddingService
    class _StubEmbeddingService:
        def __init__(self, **_):
            pass

        def get_raw_embedder(self):
            return _FakeEmbedder()

    monkeypatch.setattr(
        "backend.services.search_service.EmbeddingService",
        _StubEmbeddingService,
    )

    # Create RAG service with fake components
    fake_chat = _FakeChatService()
    rag_service = RAGService(
        corpus,
        preprocessor=_FakePreprocessor(),
        chat_service=fake_chat,
        retrieval_top_k=2,
    )

    # Execute query
    response = rag_service.query("What about Apple?", top_k=2)

    # Verify response structure
    assert isinstance(response, RAGResponse)
    assert response.question == "What about Apple?"
    assert "Answer to:" in response.answer
    assert "(with context)" in response.answer
    assert len(response.context) == 2
    assert response.metadata["num_results"] == 2

    # Verify chat service received system prompt with context
    assert fake_chat.last_system_prompt is not None
    assert "context" in fake_chat.last_system_prompt.lower()


def test_rag_service_simple_query(tmp_path, monkeypatch):
    """Test simple query interface."""
    corpus = _build_test_corpus(tmp_path)

    class _StubEmbeddingService:
        def __init__(self, **_):
            pass

        def get_raw_embedder(self):
            return _FakeEmbedder()

    monkeypatch.setattr(
        "backend.services.search_service.EmbeddingService",
        _StubEmbeddingService,
    )

    fake_chat = _FakeChatService()
    rag_service = RAGService(corpus, chat_service=fake_chat)

    answer = rag_service.query_simple("test query")

    assert isinstance(answer, str)
    assert "Answer to:" in answer


def test_rag_service_no_context_mode(tmp_path, monkeypatch):
    """Test RAG service with context disabled."""
    corpus = _build_test_corpus(tmp_path)

    class _StubEmbeddingService:
        def __init__(self, **_):
            pass

        def get_raw_embedder(self):
            return _FakeEmbedder()

    monkeypatch.setattr(
        "backend.services.search_service.EmbeddingService",
        _StubEmbeddingService,
    )

    fake_chat = _FakeChatService()
    rag_service = RAGService(corpus, chat_service=fake_chat)

    response = rag_service.query("test", include_context=False)

    # Should not include context
    assert "(with context)" not in response.answer
    assert fake_chat.last_system_prompt is None


def test_rag_service_custom_template(tmp_path, monkeypatch):
    """Test RAG service with custom context template."""
    corpus = _build_test_corpus(tmp_path)

    class _StubEmbeddingService:
        def __init__(self, **_):
            pass

        def get_raw_embedder(self):
            return _FakeEmbedder()

    monkeypatch.setattr(
        "backend.services.search_service.EmbeddingService",
        _StubEmbeddingService,
    )

    fake_chat = _FakeChatService()
    custom_template = "CUSTOM: {context}\nQ: {question}"
    rag_service = RAGService(
        corpus,
        chat_service=fake_chat,
        context_template=custom_template,
    )

    rag_service.query("test")

    assert fake_chat.last_system_prompt.startswith("CUSTOM:")


def test_rag_service_preprocessing(tmp_path, monkeypatch):
    """Test query preprocessing."""
    corpus = _build_test_corpus(tmp_path)

    class _StubEmbeddingService:
        def __init__(self, **_):
            pass

        def get_raw_embedder(self):
            return _FakeEmbedder()

    monkeypatch.setattr(
        "backend.services.search_service.EmbeddingService",
        _StubEmbeddingService,
    )

    fake_chat = _FakeChatService()
    rag_service = RAGService(
        corpus,
        preprocessor=_FakePreprocessor(),
        chat_service=fake_chat,
    )

    response = rag_service.query("  UPPERCASE QUERY  ", preprocess=True)

    # Verify preprocessing was applied
    assert response.metadata["preprocessed_query"] == "uppercase query"
