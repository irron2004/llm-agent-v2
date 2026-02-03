"""Tests for history-based routing (LLM-driven doc_lookup/history_answer)."""
import pytest
from unittest.mock import MagicMock

from backend.llm_infrastructure.llm.langgraph_agent import (
    history_answer_node,
    route_node,
    _get_last_assistant_summary,
    load_prompt_spec,
)


class TestGetLastAssistantSummary:
    """Test _get_last_assistant_summary extraction."""

    def test_extracts_last_assistant_summary(self):
        """Should extract summary from last assistant message."""
        history = [
            {"role": "user", "content": "시각적 점검 방법"},
            {"role": "assistant", "content": "시각적 점검은 장비의 외관을 확인합니다.", "summary": "시각적 점검 절차 설명"},
            {"role": "user", "content": "다음은?"},
        ]
        summary = _get_last_assistant_summary(history)
        assert summary == "시각적 점검 절차 설명"

    def test_no_summary_returns_empty(self):
        """Should return empty string if no summary field."""
        history = [
            {"role": "assistant", "content": "이것은 답변입니다."},
        ]
        summary = _get_last_assistant_summary(history)
        assert summary == ""

    def test_empty_history(self):
        """Should return empty string for empty history."""
        assert _get_last_assistant_summary([]) == ""


class TestRouteNodeLLMRouting:
    """Test route_node with LLM-based routing."""

    @pytest.fixture
    def mock_llm_retrieval(self):
        """Create a mock LLM that returns retrieval."""
        llm = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "retrieval"
        llm.generate.return_value = mock_result
        return llm

    @pytest.fixture
    def mock_llm_doc_lookup(self):
        """Create a mock LLM that returns doc_lookup."""
        llm = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "doc_lookup"
        llm.generate.return_value = mock_result
        return llm

    @pytest.fixture
    def mock_llm_general(self):
        """Create a mock LLM that returns general."""
        llm = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "general"
        llm.generate.return_value = mock_result
        return llm

    def test_rule_based_doc_id_extraction(self, mock_llm_retrieval):
        """Doc ID pattern in query should route to doc_lookup without LLM call."""
        spec = load_prompt_spec()
        state = {
            "query": "myservice 12345 설명해줘",
            "chat_history": [],
        }
        result = route_node(state, llm=mock_llm_retrieval, spec=spec)
        assert result["route"] == "doc_lookup"
        assert result["lookup_doc_ids"] == ["12345"]
        assert result["lookup_source"] == "query"
        # LLM should not be called for rule-based routing
        mock_llm_retrieval.generate.assert_not_called()

    def test_llm_returns_retrieval(self, mock_llm_retrieval):
        """LLM returns retrieval for new search queries."""
        spec = load_prompt_spec()
        state = {
            "query": "SUPRA XP 알람 해결",
            "query_en": "SUPRA XP alarm solution",
            "chat_history": [],
        }
        result = route_node(state, llm=mock_llm_retrieval, spec=spec)
        assert result["route"] == "retrieval"

    def test_llm_returns_general(self, mock_llm_general):
        """LLM returns general for casual conversation."""
        spec = load_prompt_spec()
        state = {
            "query": "안녕하세요",
            "chat_history": [],
        }
        result = route_node(state, llm=mock_llm_general, spec=spec)
        assert result["route"] == "general"

    def test_llm_doc_lookup_with_history_doc_ids(self, mock_llm_doc_lookup):
        """LLM returns doc_lookup and history has doc_ids → doc_lookup with doc_ids."""
        spec = load_prompt_spec()
        state = {
            "query": "그 문서에서 더 자세히 설명해줘",
            "query_en": "explain more from that document",
            "chat_history": [
                {"role": "assistant", "content": "답변입니다", "doc_ids": ["doc123"]},
            ],
        }
        result = route_node(state, llm=mock_llm_doc_lookup, spec=spec)
        assert result["route"] == "doc_lookup"
        assert result["lookup_doc_ids"] == ["doc123"]
        assert result["lookup_source"] == "history"

    def test_llm_doc_lookup_without_doc_ids_fallback_to_history_answer(self, mock_llm_doc_lookup):
        """LLM returns doc_lookup but no doc_ids → fallback to history_answer."""
        spec = load_prompt_spec()
        state = {
            "query": "그 다음 단계는?",
            "query_en": "what is next step?",
            "chat_history": [
                {"role": "assistant", "content": "시각적 점검을 합니다"},  # no doc_ids
            ],
        }
        result = route_node(state, llm=mock_llm_doc_lookup, spec=spec)
        assert result["route"] == "history_answer"


class TestHistoryAnswerNode:
    """Test history_answer_node behavior."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns a simple answer."""
        llm = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "다음 단계는 기능 점검입니다."
        llm.generate.return_value = mock_result
        return llm

    def test_generates_answer_from_summary(self, mock_llm):
        """Should generate answer based on last summary."""
        state = {
            "query": "그 다음 단계는?",
            "chat_history": [
                {"role": "assistant", "content": "시각적 점검을 합니다", "summary": "시각적 점검 설명"},
            ],
            "detected_language": "ko",
        }
        result = history_answer_node(state, llm=mock_llm)
        assert "answer" in result
        assert result["judge"]["faithful"] is True
        assert result["judge"]["hint"] == "history_answer"

    def test_no_summary_returns_fallback(self, mock_llm):
        """Should return fallback if no summary available."""
        state = {
            "query": "그 다음은?",
            "chat_history": [],
            "detected_language": "ko",
        }
        result = history_answer_node(state, llm=mock_llm)
        assert "이전 대화 내용을 찾을 수 없습니다" in result["answer"]
        assert result["judge"]["hint"] == "history_answer:no_summary"
