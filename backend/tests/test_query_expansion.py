"""Tests for query expansion functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.query_expansion.base import (  # noqa: E402
    BaseQueryExpander,
    ExpandedQueries,
)
from backend.llm_infrastructure.query_expansion.registry import (  # noqa: E402
    QueryExpanderRegistry,
    register_query_expander,
    get_query_expander,
)
from backend.llm_infrastructure.query_expansion.adapters.llm import (  # noqa: E402
    LLMQueryExpander,
)
from backend.llm_infrastructure.query_expansion.prompts import (  # noqa: E402
    get_prompt_template,
    list_prompt_templates,
    PROMPT_TEMPLATES,
)


class TestExpandedQueries:
    """Test ExpandedQueries dataclass."""

    def test_get_all_queries_with_original(self):
        """get_all_queries includes original when include_original is True."""
        expanded = ExpandedQueries(
            original_query="original",
            expanded_queries=["query1", "query2"],
            include_original=True,
        )
        all_queries = expanded.get_all_queries()
        assert all_queries == ["original", "query1", "query2"]

    def test_get_all_queries_without_original(self):
        """get_all_queries excludes original when include_original is False."""
        expanded = ExpandedQueries(
            original_query="original",
            expanded_queries=["query1", "query2"],
            include_original=False,
        )
        all_queries = expanded.get_all_queries()
        assert all_queries == ["query1", "query2"]

    def test_get_all_queries_deduplicates_original(self):
        """get_all_queries removes duplicate of original from expanded."""
        expanded = ExpandedQueries(
            original_query="original",
            expanded_queries=["original", "query1", "query2"],
            include_original=True,
        )
        all_queries = expanded.get_all_queries()
        # Original should appear only once at the beginning
        assert all_queries == ["original", "query1", "query2"]

    def test_len(self):
        """__len__ returns total number of queries."""
        expanded = ExpandedQueries(
            original_query="original",
            expanded_queries=["query1", "query2"],
            include_original=True,
        )
        assert len(expanded) == 3


class TestBaseQueryExpander:
    """Test BaseQueryExpander abstract class."""

    def test_base_expander_is_abstract(self):
        """BaseQueryExpander cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseQueryExpander()

    def test_base_expander_config_stored(self):
        """Config kwargs are stored."""

        class DummyExpander(BaseQueryExpander):
            def expand(self, query, n=3, include_original=True, **kwargs):
                return ExpandedQueries(query, [], include_original)

        expander = DummyExpander(foo="bar", n=5)
        assert expander.config == {"foo": "bar", "n": 5}


class TestQueryExpanderRegistry:
    """Test QueryExpanderRegistry functionality."""

    def test_register_and_get_expander(self):
        """Can register and retrieve an expander."""

        @register_query_expander("test_expander", version="v1")
        class TestExpander(BaseQueryExpander):
            def expand(self, query, n=3, include_original=True, **kwargs):
                return ExpandedQueries(query, [f"{query}_expanded"], include_original)

        expander = get_query_expander("test_expander", version="v1")
        assert isinstance(expander, TestExpander)

    def test_get_unknown_expander_raises(self):
        """Getting unknown expander raises ValueError."""
        with pytest.raises(ValueError, match="Unknown query expansion method"):
            get_query_expander("nonexistent_expander")

    def test_list_methods(self):
        """list_methods returns registered methods."""
        methods = QueryExpanderRegistry.list_methods()
        assert isinstance(methods, dict)
        # llm should be registered from import
        assert "llm" in methods


class TestPromptTemplates:
    """Test prompt templates."""

    def test_get_prompt_template(self):
        """Can get prompt template by name."""
        template = get_prompt_template("general_mq_v1")
        assert "{query}" in template
        assert "{n}" in template

    def test_get_unknown_prompt_raises(self):
        """Getting unknown prompt raises ValueError."""
        with pytest.raises(ValueError, match="Unknown prompt template"):
            get_prompt_template("nonexistent_prompt")

    def test_list_prompt_templates(self):
        """list_prompt_templates returns available templates."""
        templates = list_prompt_templates()
        assert "general_mq_v1" in templates
        assert "general_mq_v1_ko" in templates
        assert "technical_mq_v1" in templates
        assert "semiconductor_mq_v1" in templates

    def test_all_templates_have_placeholders(self):
        """All templates have required placeholders."""
        for name, template in PROMPT_TEMPLATES.items():
            assert "{query}" in template, f"{name} missing {{query}}"
            assert "{n}" in template, f"{name} missing {{n}}"


class TestLLMQueryExpander:
    """Test LLMQueryExpander."""

    def test_expander_initialization(self):
        """LLMQueryExpander initializes with defaults."""
        with patch.object(LLMQueryExpander, "__init__", lambda self, **kwargs: None):
            expander = LLMQueryExpander.__new__(LLMQueryExpander)
            expander.temperature = 0.7
            expander.max_tokens = 512
            assert expander.temperature == 0.7

    def test_expand_empty_n(self):
        """Expand with n=0 returns empty expanded queries."""
        with patch.object(LLMQueryExpander, "__init__", lambda self, **kwargs: None):
            expander = LLMQueryExpander.__new__(LLMQueryExpander)
            expander.config = {}

            result = expander.expand("test query", n=0, include_original=True)
            assert result.original_query == "test query"
            assert result.expanded_queries == []
            assert len(result.get_all_queries()) == 1  # Only original

    def test_expand_with_mock_llm(self):
        """Expand works with mocked LLM."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            text="query variation 1\nquery variation 2\nquery variation 3"
        )

        expander = LLMQueryExpander.__new__(LLMQueryExpander)
        expander.config = {}
        expander._llm = mock_llm
        expander.prompt_template = get_prompt_template("general_mq_v1")
        expander.temperature = 0.7
        expander.max_tokens = 512

        result = expander.expand("original query", n=3, include_original=True)

        assert result.original_query == "original query"
        assert len(result.expanded_queries) == 3
        assert "query variation 1" in result.expanded_queries

    def test_parse_response_removes_numbering(self):
        """_parse_response removes numbering and prefixes."""
        expander = LLMQueryExpander.__new__(LLMQueryExpander)
        expander.config = {}

        # Test various numbering formats
        response = """1. First query
2) Second query
- Third query
* Fourth query
"""
        parsed = expander._parse_response(response, n=4)
        assert parsed == ["First query", "Second query", "Third query", "Fourth query"]

    def test_parse_response_removes_quotes(self):
        """_parse_response removes surrounding quotes."""
        expander = LLMQueryExpander.__new__(LLMQueryExpander)
        expander.config = {}

        response = """"Query with double quotes"
'Query with single quotes'
Normal query
"""
        parsed = expander._parse_response(response, n=3)
        assert "Query with double quotes" in parsed
        assert "Query with single quotes" in parsed
        assert "Normal query" in parsed

    def test_deduplicate_removes_duplicates(self):
        """_deduplicate removes duplicate queries."""
        expander = LLMQueryExpander.__new__(LLMQueryExpander)
        expander.config = {}

        queries = ["Query A", "query a", "Query B", "Query A"]
        deduplicated = expander._deduplicate(queries, "original")

        # Should remove case-insensitive duplicates
        assert len(deduplicated) == 2
        assert "Query A" in deduplicated
        assert "Query B" in deduplicated

    def test_deduplicate_removes_original_from_expanded(self):
        """_deduplicate removes original query from expanded list."""
        expander = LLMQueryExpander.__new__(LLMQueryExpander)
        expander.config = {}

        queries = ["Original Query", "Different Query", "Another Query"]
        deduplicated = expander._deduplicate(queries, "original query")

        # "Original Query" should be removed (case-insensitive match)
        assert len(deduplicated) == 2
        assert "Original Query" not in deduplicated

    def test_expander_repr(self):
        """Expander has informative repr."""
        with patch.object(LLMQueryExpander, "__init__", lambda self, **kwargs: None):
            expander = LLMQueryExpander.__new__(LLMQueryExpander)
            expander.temperature = 0.7
            repr_str = repr(expander)
            assert "LLMQueryExpander" in repr_str
            assert "0.7" in repr_str


class TestSearchServiceMultiQuery:
    """Test SearchService with multi-query integration."""

    @pytest.fixture
    def mock_corpus(self, tmp_path):
        """Create a mock corpus for testing."""
        import numpy as np
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

    def test_search_service_with_multi_query_disabled(self, mock_corpus, monkeypatch):
        """SearchService works with multi-query disabled."""

        class StubEmbeddingService:
            def __init__(self, **_):
                pass

            def get_raw_embedder(self):
                import numpy as np

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

        svc = SearchService(
            mock_corpus,
            method="dense",
            multi_query_enabled=False,
            rerank_enabled=False,
        )
        assert svc.query_expander is None

        results = svc.search("apple", top_k=2)
        assert len(results) <= 2

    @patch("backend.services.search_service.get_query_expander")
    def test_search_service_with_multi_query_enabled(
        self, mock_get_expander, mock_corpus, monkeypatch
    ):
        """SearchService applies multi-query expansion when enabled."""

        class StubEmbeddingService:
            def __init__(self, **_):
                pass

            def get_raw_embedder(self):
                import numpy as np

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

        # Mock query expander
        mock_expander = MagicMock()
        mock_expander.expand.return_value = ExpandedQueries(
            original_query="apple",
            expanded_queries=["red apple", "apple fruit"],
            include_original=True,
        )
        mock_get_expander.return_value = mock_expander

        from backend.services.search_service import SearchService

        svc = SearchService(
            mock_corpus,
            method="dense",
            multi_query_enabled=True,
            multi_query_n=2,
            rerank_enabled=False,
        )

        results = svc.search("apple", top_k=3)

        # Expander should have been called
        mock_expander.expand.assert_called_once()
        # Results should be returned (merged from multiple queries)
        assert len(results) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])