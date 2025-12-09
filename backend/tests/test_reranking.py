"""Tests for reranking functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.retrieval.base import RetrievalResult  # noqa: E402
from backend.llm_infrastructure.reranking.base import BaseReranker  # noqa: E402
from backend.llm_infrastructure.reranking.registry import (  # noqa: E402
    RerankerRegistry,
    register_reranker,
    get_reranker,
)
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (  # noqa: E402
    CrossEncoderReranker,
)


class TestBaseReranker:
    """Test BaseReranker abstract class."""

    def test_base_reranker_is_abstract(self):
        """BaseReranker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseReranker()

    def test_base_reranker_config_stored(self):
        """Config kwargs are stored."""

        class DummyReranker(BaseReranker):
            def rerank(self, query, results, top_k=None, **kwargs):
                return results

        reranker = DummyReranker(foo="bar", batch_size=32)
        assert reranker.config == {"foo": "bar", "batch_size": 32}


class TestRerankerRegistry:
    """Test RerankerRegistry functionality."""

    def test_register_and_get_reranker(self):
        """Can register and retrieve a reranker."""

        @register_reranker("test_reranker", version="v1")
        class TestReranker(BaseReranker):
            def rerank(self, query, results, top_k=None, **kwargs):
                return results

        reranker = get_reranker("test_reranker", version="v1")
        assert isinstance(reranker, TestReranker)

    def test_get_unknown_reranker_raises(self):
        """Getting unknown reranker raises ValueError."""
        with pytest.raises(ValueError, match="Unknown reranking method"):
            get_reranker("nonexistent_reranker")

    def test_list_methods(self):
        """list_methods returns registered methods."""
        methods = RerankerRegistry.list_methods()
        assert isinstance(methods, dict)
        # cross_encoder should be registered from import
        assert "cross_encoder" in methods


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker."""

    @pytest.fixture
    def mock_results(self) -> list[RetrievalResult]:
        """Create mock retrieval results."""
        return [
            RetrievalResult(
                doc_id="doc1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.8,
                raw_text="Machine learning is a subset of artificial intelligence.",
            ),
            RetrievalResult(
                doc_id="doc2",
                content="Deep learning uses neural networks.",
                score=0.7,
                raw_text="Deep learning uses neural networks.",
            ),
            RetrievalResult(
                doc_id="doc3",
                content="Python is a programming language.",
                score=0.6,
                raw_text="Python is a programming language.",
            ),
        ]

    def test_reranker_initialization(self):
        """CrossEncoderReranker initializes with defaults."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == CrossEncoderReranker.DEFAULT_MODEL
        assert reranker.device == "cpu"
        assert reranker._model is None  # Lazy loading

    def test_reranker_custom_model(self):
        """CrossEncoderReranker accepts custom model."""
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            device="cuda",
            batch_size=16,
        )
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert reranker.device == "cuda"
        assert reranker.batch_size == 16

    def test_rerank_empty_results(self):
        """Reranking empty results returns empty list."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [])
        assert results == []

    @patch.object(CrossEncoderReranker, "_load_model")
    def test_rerank_with_mock_model(self, mock_load_model, mock_results):
        """Reranking works with mocked cross-encoder."""
        # Mock the cross-encoder model
        mock_model = MagicMock()
        # Return scores that reorder: doc3 > doc1 > doc2
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.9])
        mock_load_model.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("What is AI?", mock_results)

        # Check order changed based on new scores
        assert len(reranked) == 3
        assert reranked[0].doc_id == "doc3"  # Highest score (0.9)
        assert reranked[1].doc_id == "doc1"  # Second (0.5)
        assert reranked[2].doc_id == "doc2"  # Third (0.3)

        # Check scores are updated
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.5
        assert reranked[2].score == 0.3

        # Check original score preserved in metadata
        assert reranked[0].metadata["original_score"] == 0.6
        assert reranked[0].metadata["rerank_model"] == CrossEncoderReranker.DEFAULT_MODEL

    @patch.object(CrossEncoderReranker, "_load_model")
    def test_rerank_with_top_k(self, mock_load_model, mock_results):
        """Reranking with top_k limits results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.9])
        mock_load_model.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("What is AI?", mock_results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].doc_id == "doc3"
        assert reranked[1].doc_id == "doc1"

    @patch.object(CrossEncoderReranker, "_load_model")
    def test_rerank_uses_raw_text_when_available(self, mock_load_model, mock_results):
        """Reranking uses raw_text for scoring when available."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.9])
        mock_load_model.return_value = mock_model

        reranker = CrossEncoderReranker()
        reranker.rerank("test query", mock_results)

        # Check that predict was called with raw_text
        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert pairs[0][1] == mock_results[0].raw_text

    def test_reranker_repr(self):
        """Reranker has informative repr."""
        reranker = CrossEncoderReranker(model_name="test-model", device="cuda")
        repr_str = repr(reranker)
        assert "CrossEncoderReranker" in repr_str
        assert "test-model" in repr_str
        assert "cuda" in repr_str


class TestSearchServiceWithReranking:
    """Test SearchService with reranking integration."""

    @pytest.fixture
    def mock_corpus(self, tmp_path):
        """Create a mock corpus for testing."""
        from backend.services.document_service import (
            DocumentIndexService,
            SourceDocument,
        )

        class FakeEmbedder:
            def embed(self, text):
                return np.array([len(text), text.count("a")], dtype=np.float32)

            def embed_batch(self, texts):
                return np.vstack([self.embed(t) for t in texts])

        docs = [
            SourceDocument(doc_id="d1", text="apple banana cherry"),
            SourceDocument(doc_id="d2", text="banana cherry date"),
            SourceDocument(doc_id="d3", text="cherry date elderberry"),
        ]
        svc = DocumentIndexService(embedder=FakeEmbedder())
        return svc.index(docs, preprocess=False, persist_dir=tmp_path)

    def test_search_service_with_reranking_disabled(self, mock_corpus, monkeypatch):
        """SearchService works with reranking disabled."""

        class StubEmbeddingService:
            def __init__(self, **_):
                pass

            def get_raw_embedder(self):
                class FakeEmbedder:
                    def embed(self, text):
                        return np.array([len(text), text.count("a")], dtype=np.float32)

                    def embed_batch(self, texts):
                        return np.vstack([self.embed(t) for t in texts])

                return FakeEmbedder()

        monkeypatch.setattr(
            "backend.services.search_service.EmbeddingService",
            StubEmbeddingService,
        )

        from backend.services.search_service import SearchService

        svc = SearchService(mock_corpus, method="dense", rerank_enabled=False)
        assert svc.reranker is None

        results = svc.search("apple", top_k=2)
        assert len(results) == 2

    @patch("backend.services.search_service.get_reranker")
    def test_search_service_with_reranking_enabled(
        self, mock_get_reranker, mock_corpus, monkeypatch
    ):
        """SearchService applies reranking when enabled."""

        class StubEmbeddingService:
            def __init__(self, **_):
                pass

            def get_raw_embedder(self):
                class FakeEmbedder:
                    def embed(self, text):
                        return np.array([len(text), text.count("a")], dtype=np.float32)

                    def embed_batch(self, texts):
                        return np.vstack([self.embed(t) for t in texts])

                return FakeEmbedder()

        monkeypatch.setattr(
            "backend.services.search_service.EmbeddingService",
            StubEmbeddingService,
        )

        # Mock reranker that reverses results
        mock_reranker = MagicMock()

        def mock_rerank(query, results, top_k=None):
            reversed_results = list(reversed(results))
            if top_k:
                reversed_results = reversed_results[:top_k]
            return reversed_results

        mock_reranker.rerank.side_effect = mock_rerank
        mock_get_reranker.return_value = mock_reranker

        from backend.services.search_service import SearchService

        svc = SearchService(
            mock_corpus,
            method="dense",
            rerank_enabled=True,
            rerank_method="cross_encoder",
            rerank_top_k=2,
        )

        assert svc.reranker is mock_reranker

        results = svc.search("apple", top_k=3)
        # Reranker should have been called
        mock_reranker.rerank.assert_called_once()

    @patch("backend.services.search_service.get_reranker")
    def test_search_service_rerank_override(
        self, mock_get_reranker, mock_corpus, monkeypatch
    ):
        """SearchService rerank parameter can override service setting."""

        class StubEmbeddingService:
            def __init__(self, **_):
                pass

            def get_raw_embedder(self):
                class FakeEmbedder:
                    def embed(self, text):
                        return np.array([len(text), text.count("a")], dtype=np.float32)

                    def embed_batch(self, texts):
                        return np.vstack([self.embed(t) for t in texts])

                return FakeEmbedder()

        monkeypatch.setattr(
            "backend.services.search_service.EmbeddingService",
            StubEmbeddingService,
        )

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        mock_get_reranker.return_value = mock_reranker

        from backend.services.search_service import SearchService

        # Service has reranking enabled
        svc = SearchService(mock_corpus, method="dense", rerank_enabled=True)

        # But we override to disable it for this search
        svc.search("apple", rerank=False)
        mock_reranker.rerank.assert_not_called()


@pytest.mark.integration
class TestCrossEncoderRerankerIntegration:
    """Integration tests for CrossEncoderReranker (requires model download)."""

    @pytest.fixture
    def mock_results(self) -> list[RetrievalResult]:
        """Create mock retrieval results for integration test."""
        return [
            RetrievalResult(
                doc_id="doc1",
                content="The capital of France is Paris.",
                score=0.5,
                raw_text="The capital of France is Paris.",
            ),
            RetrievalResult(
                doc_id="doc2",
                content="Berlin is a city in Germany.",
                score=0.8,
                raw_text="Berlin is a city in Germany.",
            ),
            RetrievalResult(
                doc_id="doc3",
                content="Paris is known for the Eiffel Tower.",
                score=0.3,
                raw_text="Paris is known for the Eiffel Tower.",
            ),
        ]

    @pytest.mark.slow
    def test_cross_encoder_reranks_correctly(self, mock_results):
        """Cross-encoder reranks results based on query relevance."""
        reranker = CrossEncoderReranker(device="cpu")
        query = "What is the capital of France?"

        reranked = reranker.rerank(query, mock_results, top_k=2)

        # doc1 and doc3 should be ranked higher for France-related query
        assert len(reranked) == 2
        top_doc_ids = {r.doc_id for r in reranked}
        assert "doc1" in top_doc_ids or "doc3" in top_doc_ids
        # doc2 about Berlin should be ranked lower
        assert reranked[0].doc_id != "doc2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])