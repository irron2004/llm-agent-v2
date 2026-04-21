"""Regression test: general_ans_v2 and ts_ans_v2 handle empty REFS with clarifying questions.

Verifies (updated 2026-04-21 for chat-log alignment patch):
- REFS empty → explicit "not found" + (optional) 1-2 clarifying questions, no hallucination
- Question-type awareness → prompt mentions inquiry-style AND procedure-style structures
- No hallucination → prompt forbids evidence outside REFS
- Korean-only answer mandate
- Citation from REFS only

Phrasing was intentionally loosened so that the reference service's response patterns
(adaptive structure, tables, multi-citation) can be reproduced while preserving the
"REFS-only evidence" guard. These tests verify the semantic invariants rather than
the exact legacy wording.
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


def _has_empty_refs_handling(system: str) -> bool:
    """새 프롬프트는 `REFS 가 완전히 비어있거나` 또는 `제공된 참고자료에는` 문구로 empty REFS 를 처리."""
    return (
        "REFS 가 완전히 비어있거나" in system
        or "REFS가 비어있거나" in system
        or "제공된 참고자료에는" in system
    )


def _has_not_found_phrase(system: str) -> bool:
    return (
        "제공된 참고자료에는" in system
        or "찾지 못했습니다" in system
        or "정보가 없습니다" in system
    )


def _has_clarifying_option(system: str) -> bool:
    return "확인 질문" in system


def _forbids_fabrication(system: str) -> bool:
    return (
        "추측" in system
        or "추측하거나" in system
        or "절대로" in system
        or "발명" in system
    )


def test_general_ans_v2_system_prompt_contains_empty_refs_clarification() -> None:
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system

    assert _has_empty_refs_handling(system)
    assert _has_not_found_phrase(system)
    assert _has_clarifying_option(system)
    assert _forbids_fabrication(system)


def test_ts_ans_v2_system_prompt_contains_empty_refs_clarification() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system

    assert _has_empty_refs_handling(system)
    assert _has_not_found_phrase(system)
    assert _has_clarifying_option(system)
    assert _forbids_fabrication(system)


def test_general_ans_v2_distinguishes_inquiry_vs_procedure() -> None:
    """새 프롬프트는 질문 유형을 조회/절차 2분법 대신 6종(스펙/이력/트러블슈팅/절차/조회/체크리스트)으로 분류.

    이전의 '발명하지 마세요' 부정형 문구 대신 'REFS 에 있는 내용만' / '(참고)' 긍정형으로 유도.
    따라서 invariants 로 검증: 조회 계열 + 절차 계열이 모두 system 에 언급되고,
    식별자 fabrication 금지가 유지되는지만 확인.
    """
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system

    # 조회 계열: '조회' / '스펙' / '위치' / '이력' 중 최소 하나
    assert any(kw in system for kw in ("조회", "스펙", "위치", "이력"))
    # 절차 계열: '절차' / '작업 절차' 중 최소 하나
    assert "절차" in system
    # REFS 기반 강제
    assert "REFS" in system
    # 식별자 fabrication 금지
    assert (
        "USER QUESTION 에 없는" in system
        or "USER QUESTION에 없는" in system
        or "식별자" in system
    )


def test_ts_ans_v2_distinguishes_inquiry_vs_procedure() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system

    assert any(kw in system for kw in ("조회", "스펙", "위치", "이력", "현상", "증상"))
    assert "절차" in system
    assert "REFS" in system
    assert (
        "USER QUESTION 에 없는" in system
        or "USER QUESTION에 없는" in system
        or "식별자" in system
    )


def test_general_ans_v2_forbids_hallucination_in_empty_refs() -> None:
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system
    assert _forbids_fabrication(system)


def test_ts_ans_v2_forbids_hallucination_in_empty_refs() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system
    assert _forbids_fabrication(system)


def test_general_ans_v2_uses_korean_only() -> None:
    spec = load_prompt_spec("v2")
    assert "한국어로 답변" in spec.general_ans.system or "한국어" in spec.general_ans.system


def test_ts_ans_v2_uses_korean_only() -> None:
    spec = load_prompt_spec("v2")
    assert "한국어로 답변" in spec.ts_ans.system or "한국어" in spec.ts_ans.system


def test_general_ans_v2_cites_only_refs() -> None:
    spec = load_prompt_spec("v2")
    system = spec.general_ans.system
    assert "REFS" in system
    assert "증거" in system or "인용" in system


def test_ts_ans_v2_cites_only_refs() -> None:
    spec = load_prompt_spec("v2")
    system = spec.ts_ans.system
    assert "REFS" in system
    assert "증거" in system or "인용" in system


def test_answer_node_with_empty_refs_receives_clarification_instruction() -> None:
    """answer_node 가 general route 에서 empty REFS 시 system prompt 에 empty-REFS 처리 지침을 넘기는지.

    Phase A (2026-04-21) 로 general/ts 경로는 strict SOP 템플릿 검증(`## 작업 절차` 강제)을
    거치지 않으므로 [FORMAT FIX] 재시도가 발생하지 않아야 한다. 따라서 last_system 은
    base system prompt 그대로여야 하고, empty-REFS 처리 문구와 clarifying-question 문구가 모두 존재해야 한다.
    """
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
    # general route 에는 FORMAT FIX 가 더 이상 붙으면 안 됨
    assert "[FORMAT FIX]" not in llm.last_system, (
        "general route 는 strict SOP 템플릿 검증을 건너뛰어야 하는데 FORMAT FIX 가 붙었다."
    )
    assert _has_empty_refs_handling(llm.last_system)
    assert _has_clarifying_option(llm.last_system)
