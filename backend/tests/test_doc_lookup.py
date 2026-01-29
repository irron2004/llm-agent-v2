"""doc_lookup 관련 테스트."""

import pytest
from unittest.mock import MagicMock, patch

from backend.llm_infrastructure.llm.langgraph_agent import (
    _extract_doc_id_from_query,
    _format_history_for_prompt,
    _get_doc_ids_from_history,
    doc_lookup_node,
    ChatHistoryEntry,
)


class TestExtractDocIdFromQuery:
    """Rule 기반 doc_id 추출 테스트."""

    def test_myservice_with_space(self):
        """myservice 29392 패턴."""
        result = _extract_doc_id_from_query("myservice 29392 설명해줘")
        assert result == ("myservice", "29392")

    def test_myservice_with_dash(self):
        """myservice-29392 패턴."""
        result = _extract_doc_id_from_query("myservice-29392에 대해")
        assert result == ("myservice", "29392")

    def test_myservice_with_underscore(self):
        """myservice_29392 패턴."""
        result = _extract_doc_id_from_query("myservice_12345 내용")
        assert result == ("myservice", "12345")

    def test_myservice_attached(self):
        """myservice29392 패턴 (붙여쓰기)."""
        result = _extract_doc_id_from_query("myservice29392 알려줘")
        assert result == ("myservice", "29392")

    def test_gcb_pattern(self):
        """gcb 12345 패턴."""
        result = _extract_doc_id_from_query("gcb 12345 조회")
        assert result == ("gcb", "12345")

    def test_sop_pattern(self):
        """sop-001 패턴."""
        result = _extract_doc_id_from_query("sop-001 확인해줘")
        assert result == ("sop", "001")

    def test_case_insensitive(self):
        """대소문자 구분 없음."""
        result = _extract_doc_id_from_query("MYSERVICE 99999")
        assert result == ("myservice", "99999")

    def test_no_match(self):
        """매칭 실패 시 None."""
        result = _extract_doc_id_from_query("SUPRA XP 센서 이상")
        assert result is None

    def test_no_match_partial(self):
        """부분 매칭 불가."""
        result = _extract_doc_id_from_query("service 12345")  # myservice 아님
        assert result is None


class TestFormatHistoryForPrompt:
    """history 포맷팅 테스트."""

    def test_empty_history(self):
        """빈 히스토리."""
        result = _format_history_for_prompt([])
        assert result == ""

    def test_user_entry(self):
        """user 항목."""
        history = [{"role": "user", "content": "SUPRA XP 센서 이상"}]
        result = _format_history_for_prompt(history)
        assert "User: SUPRA XP 센서 이상" in result

    def test_assistant_entry(self):
        """assistant 항목."""
        history = [
            {
                "role": "assistant",
                "summary": "센서 이상 시 캘리브레이션 확인",
                "refs": ["SUPRA XP TS", "Calibration SOP"],
                "doc_ids": ["12345", "67890"],
            }
        ]
        result = _format_history_for_prompt(history)
        assert "Assistant: 센서 이상 시 캘리브레이션 확인" in result
        assert "SUPRA XP TS" in result

    def test_multi_turn(self):
        """다중 턴."""
        history = [
            {"role": "user", "content": "첫 번째 질문"},
            {"role": "assistant", "summary": "첫 번째 답변", "refs": [], "doc_ids": []},
            {"role": "user", "content": "두 번째 질문"},
        ]
        result = _format_history_for_prompt(history)
        assert "User: 첫 번째 질문" in result
        assert "User: 두 번째 질문" in result
        assert "Assistant: 첫 번째 답변" in result

    def test_max_5_turns(self):
        """최근 5턴만 사용."""
        history = [{"role": "user", "content": f"질문 {i}"} for i in range(10)]
        result = _format_history_for_prompt(history)
        # 최근 5개만 포함
        assert "질문 5" in result
        assert "질문 9" in result
        assert "질문 0" not in result


class TestGetDocIdsFromHistory:
    """history에서 doc_ids 추출 테스트."""

    def test_get_most_recent(self):
        """가장 최근 assistant의 doc_ids."""
        history = [
            {"role": "assistant", "doc_ids": ["old_1", "old_2"]},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "doc_ids": ["new_1", "new_2"]},
        ]
        result = _get_doc_ids_from_history(history)
        assert result == ["new_1", "new_2"]

    def test_max_3_doc_ids(self):
        """최대 3개만 반환."""
        history = [
            {"role": "assistant", "doc_ids": ["d1", "d2", "d3", "d4", "d5"]},
        ]
        result = _get_doc_ids_from_history(history)
        assert result == ["d1", "d2", "d3"]

    def test_empty_when_no_doc_ids(self):
        """doc_ids 없으면 빈 리스트."""
        history = [{"role": "assistant", "summary": "답변만 있음"}]
        result = _get_doc_ids_from_history(history)
        assert result == []

    def test_empty_when_no_assistant(self):
        """assistant 없으면 빈 리스트."""
        history = [{"role": "user", "content": "질문만"}]
        result = _get_doc_ids_from_history(history)
        assert result == []


class TestDocLookupNode:
    """doc_lookup_node 테스트."""

    def test_success(self):
        """문서 조회 성공."""
        mock_fetcher = MagicMock(return_value=[{"doc_id": "12345", "content": "test"}])
        state = {"lookup_doc_ids": ["12345"], "lookup_source": "query"}

        result = doc_lookup_node(state, doc_fetcher=mock_fetcher)

        assert "docs" in result
        assert len(result["docs"]) == 1
        assert result["lookup_doc_ids"] == ["12345"]
        mock_fetcher.assert_called_once_with("12345")

    def test_fallback_no_doc_ids(self):
        """doc_ids 없으면 fallback."""
        mock_fetcher = MagicMock()
        state = {"lookup_doc_ids": [], "lookup_source": "query"}

        result = doc_lookup_node(state, doc_fetcher=mock_fetcher)

        assert result["route"] == "general"
        mock_fetcher.assert_not_called()

    def test_fallback_invalid_doc_id(self):
        """doc_id 검증 실패 시 fallback."""
        mock_fetcher = MagicMock(return_value=[])  # 문서 없음
        state = {"lookup_doc_ids": ["invalid_id"], "lookup_source": "query"}

        result = doc_lookup_node(state, doc_fetcher=mock_fetcher)

        assert result["route"] == "general"

    def test_partial_success(self):
        """일부 doc_id만 유효."""
        def mock_fetch(doc_id):
            if doc_id == "valid":
                return [{"doc_id": "valid", "content": "test"}]
            return []

        state = {"lookup_doc_ids": ["valid", "invalid"], "lookup_source": "history"}

        result = doc_lookup_node(state, doc_fetcher=mock_fetch)

        assert "docs" in result
        assert len(result["docs"]) == 1
        assert result["lookup_doc_ids"] == ["valid"]

    def test_max_3_doc_ids(self):
        """최대 3개 doc_id만 조회."""
        call_count = {"count": 0}

        def mock_fetch(doc_id):
            call_count["count"] += 1
            return [{"doc_id": doc_id, "content": "test"}]

        state = {"lookup_doc_ids": ["d1", "d2", "d3", "d4", "d5"], "lookup_source": "query"}

        doc_lookup_node(state, doc_fetcher=mock_fetch)

        assert call_count["count"] == 3  # 최대 3번만 호출
