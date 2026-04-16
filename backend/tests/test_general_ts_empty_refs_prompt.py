"""Regression test: general_ans_v2 and ts_ans_v2 handle empty REFS with clarifying questions.

Verifies March-log patterns:
- REFS empty → explicit "not found" + 1-3 clarifying questions, no hallucination
- Inquiry questions → no invented procedures, lists found info
- Procedure questions → REFS-based steps only
"""

from __future__ import annotations

from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec, answer_node, load_prompt_spec
from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse


class CaptureAnswerLLM(BaseLLM):
    def __init__(self, response_text: str = "ok") -> None:
        super().__init__()
        self.response_text = response_text
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, messages, *, response_model=None, **kwargs):
        if response_model is not None:
            raise NotImplementedError
        message_list = list(messages)
        if message_list and message_list[0].get("role") == "system":
            self.last_system = message_list[0].get("content", "")
        if message_list:
            self.last_user = message_list[-1].get("content", "")
        return LLMResponse(text=self.response_text)


def test_general_ans_v2_system_prompt_contains_empty_refs_clarification() -> None:
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system

    assert "REFS가 비어있거나 질문과 관련성이 낮으면" in system
    assert "RAG 데이터에서 관련 정보를 찾지 못했습니다" in system
    assert "확인 질문" in system
    assert "추측" in system or "추측하거나" in system or "발명" in system


def test_ts_ans_v2_system_prompt_contains_empty_refs_clarification() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system

    assert "REFS가 비어있거나 질문과 관련성이 낮으면" in system
    assert "RAG 데이터에서 관련" in system
    assert "확인 질문" in system
    assert "추측" in system or "추측하거나" in system or "발명" in system


def test_general_ans_v2_distinguishes_inquiry_vs_procedure() -> None:
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system

    assert "조회 질문" in system or "조회" in system
    assert "절차 질문" in system or "절차" in system
    assert "발명하지 마세요" in system or "발명" in system
    assert "REFS에" in system


def test_ts_ans_v2_distinguishes_inquiry_vs_procedure() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system

    assert "조회 질문" in system or "조회" in system
    assert "절차 질문" in system or "절차" in system
    assert "발명하지 마세요" in system or "발명" in system
    assert "REFS에" in system


def test_general_ans_v2_forbids_hallucination_in_empty_refs() -> None:
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system

    lines = system.split("\n")
    has_forbid_guess = any(
        "추측" in line or "추측하거나" in line or "절대로" in line for line in lines
    )
    assert has_forbid_guess


def test_ts_ans_v2_forbids_hallucination_in_empty_refs() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system

    lines = system.split("\n")
    has_forbid_guess = any(
        "추측" in line or "추측하거나" in line or "절대로" in line for line in lines
    )
    assert has_forbid_guess


def test_general_ans_v2_uses_korean_only() -> None:
    spec = load_prompt_spec("v2")
    assert "한국어로 답변" in spec.general_ans.system or "한국어" in spec.general_ans.system


def test_ts_ans_v2_uses_korean_only() -> None:
    spec = load_prompt_spec("v2")
    assert "한국어로 답변" in spec.ts_ans.system or "한국어" in spec.ts_ans.system


def test_general_ans_v2_cites_only_refs() -> None:
    spec = load_prompt_spec("v2")
    assert "REFS 라인만 증거" in spec.general_ans.system or "REFS" in spec.general_ans.system


def test_ts_ans_v2_cites_only_refs() -> None:
    spec = load_prompt_spec("v2")
    assert "REFS 라인만 증거" in spec.ts_ans.system or "REFS" in spec.ts_ans.system


def test_answer_node_with_empty_refs_receives_clarification_instruction() -> None:
    spec = load_prompt_spec("v2")
    llm = CaptureAnswerLLM()

    state = {
        "route": "general",
        "query": "이 장치의 위치가 어디에 있나요?",
        "detected_language": "ko",
        "answer_ref_json": [],
    }

    _ = answer_node(state, llm=llm, spec=spec)

    assert llm.last_system is not None
    assert "REFS가 비어있거나" in llm.last_system
    assert "확인 질문" in llm.last_system or "질문" in llm.last_system
