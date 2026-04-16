"""Test empty-REFS guard in judge_node."""

from unittest.mock import MagicMock

import pytest

from llm_infrastructure.llm.langgraph_agent import judge_node


class FailingLLM:
    """Mock LLM that raises if called — guarantees short-circuit path was taken."""

    def generate(self, *args, **kwargs):
        raise RuntimeError(
            "LLM should not be called when refs are EMPTY and answer has clarifying question"
        )


@pytest.fixture
def failing_llm():
    return FailingLLM()


@pytest.fixture
def mock_spec():
    spec = MagicMock()
    spec.judge_setup_sys = "setup"
    spec.judge_ts_sys = "ts"
    spec.judge_general_sys = "general"
    return spec


@pytest.mark.parametrize(
    "route,answer,expected_faithful,expected_hint",
    [
        # (route, answer, faithful, hint)
        # empty refs + clarifying question → faithful
        (
            "general",
            "해당 정보를 찾지 못했습니다. 어떤 종류의 Chamber를 사용하시나요?",
            True,
            "no_refs: clarification",
        ),
        ("ts", "찾지 못했습니다. 더 구체적인 장치는 무엇인가요?", True, "no_refs: clarification"),
        (
            "setup",
            "검색 결과가 없습니다. PM 문서를特定할 수 있나요?",
            True,
            "no_refs: clarification",
        ),
        # empty refs + no question → unfaithful
        ("general", "찾지 못했습니다.", False, "no refs to evaluate"),
        ("ts", "검색 결과가 없습니다.", False, "no refs to evaluate"),
        # empty refs + with citation → unfaithful
        ("general", "찾지 못했습니다. [1]을 참조하세요.", False, "no refs to evaluate"),
        ("ts", "찾지 못했습니다. 특정 Chamber가 없습니다[1].", False, "no refs to evaluate"),
        # not-empty refs → normal behavior (LLM should be called)
        # These cases are covered by the non-empty refs flow
    ],
)
def test_judge_node_empty_refs_guard(
    failing_llm, mock_spec, route, answer, expected_faithful, expected_hint
):
    """Test empty-REFS guard short-circuits without calling LLM."""
    state = {
        "route": route,
        "query": "테스트 질문",
        "answer": answer,
        "ref_json": [],
        "answer_ref_json": [],
    }
    result = judge_node(state, llm=failing_llm, spec=mock_spec)
    judge = result["judge"]
    assert judge["faithful"] == expected_faithful, (
        f"Expected faithful={expected_faithful}, got {judge}"
    )
    assert expected_hint in judge.get("hint", ""), (
        f"Expected hint to contain '{expected_hint}', got '{judge.get('hint')}'"
    )


def test_judge_node_empty_refs_with_clarification_multiple_questions(failing_llm, mock_spec):
    """Test empty-REFS guard accepts up to 3 clarifying questions."""
    answer = """검색 결과가 없습니다.
    1. Chamber 용량이 어떻게 되나요?
    2._alarm 종류가 뭐예요?
    3.장치Vendor가哪家인가요?"""
    state = {
        "route": "general",
        "query": "PM 문서 검색",
        "answer": answer,
        "ref_json": [],
        "answer_ref_json": [],
    }
    result = judge_node(state, llm=failing_llm, spec=mock_spec)
    judge = result["judge"]
    assert judge["faithful"] is True
    assert "no_refs: clarification" in judge["hint"]


def test_judge_node_non_empty_refs_calls_llm(mock_spec):
    """Test that non-empty refs still calls LLM (guard does not trigger)."""
    call_tracker = {"called": False}

    class TrackingLLM:
        def generate(self, *args, **kwargs):
            call_tracker["called"] = True

            class Out:
                text = '{"faithful": true, "issues": [], "hint": "ok"}'

            return Out()

    llm = TrackingLLM()
    state = {
        "route": "general",
        "query": "테스트 질문",
        "answer": "Some answer",
        "ref_json": [{"rank": 1, "doc_id": "doc1", "content": "some content"}],
        "answer_ref_json": [],
    }
    result = judge_node(state, llm=llm, spec=mock_spec)
    assert call_tracker["called"], "LLM should be called when refs are not empty"
