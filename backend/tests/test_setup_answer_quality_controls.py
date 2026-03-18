from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec, answer_node, load_prompt_spec
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate


class CaptureAnswerLLM(BaseLLM):
    def __init__(self, response_text: str = "ok") -> None:
        super().__init__()
        self.response_text = response_text
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_kwargs: dict[str, Any] = {}

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_model is not None:
            raise NotImplementedError
        message_list = list(messages)
        if message_list and message_list[0].get("role") == "system":
            self.last_system = message_list[0].get("content", "")
        if message_list:
            self.last_user = message_list[-1].get("content", "")
        self.last_kwargs = dict(kwargs)
        return LLMResponse(text=self.response_text)


def _prompt(name: str, system: str = "", user: str = "{sys.query}\n{ref_text}") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system=system, user=user, raw={})


def _spec() -> PromptSpec:
    empty = _prompt("empty")
    return PromptSpec(
        router=empty,
        setup_mq=empty,
        ts_mq=empty,
        general_mq=empty,
        st_gate=empty,
        st_mq=empty,
        setup_ans=_prompt(
            "setup_ans", system="KO_SETUP", user="질문:{sys.query}\nREFS:\n{ref_text}"
        ),
        ts_ans=empty,
        general_ans=empty,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
    )


def test_answer_node_prioritizes_procedure_refs_for_setup_route() -> None:
    llm = CaptureAnswerLLM()
    spec = _spec()
    state = {
        "route": "setup",
        "query": "EFEM PIO SENSOR BOARD 교체 절차",
        "detected_language": "ko",
        "answer_ref_json": [
            {
                "rank": 1,
                "doc_id": "doc_scope",
                "content": "Scope and overview information.",
                "metadata": {"device_name": "ZEDIUS XP"},
            },
            {
                "rank": 2,
                "doc_id": "doc_procedure",
                "content": "Work Procedure: remove cover, replace board, verify signals.",
                "metadata": {"device_name": "ZEDIUS XP"},
            },
        ],
    }

    _ = answer_node(state, llm=llm, spec=spec)

    assert llm.last_user is not None
    # doc_section 그룹핑으로 procedure 문서만 선택되고 scope 문서는 제외됨
    assert "doc_procedure" in llm.last_user
    assert "Work Procedure" in llm.last_user


def test_answer_node_uses_lower_temperature_for_setup_route() -> None:
    llm = CaptureAnswerLLM()
    spec = _spec()
    state = {
        "route": "setup",
        "query": "교체 절차",
        "detected_language": "ko",
        "answer_ref_json": [],
    }

    _ = answer_node(state, llm=llm, spec=spec)

    assert llm.last_kwargs.get("temperature") == 0.2


def test_answer_node_postprocesses_setup_artifacts() -> None:
    llm = CaptureAnswerLLM(
        response_text=(
            "| 단계 | 내용 |\n|---|---|\n| 1️⃣ | 점검 후 교체 【[1] p2】 EFIM [...] REFS TBD |\n"
        )
    )
    spec = _spec()
    state = {
        "route": "setup",
        "query": "교체 절차",
        "detected_language": "ko",
        "answer_ref_json": [],
    }

    result = answer_node(state, llm=llm, spec=spec)
    answer = str(result["answer"])

    assert "1️⃣" not in answer
    assert "【1】" not in answer
    assert "【[1] p2】" not in answer
    assert "EFIM" not in answer
    assert "[...]" not in answer
    assert "|---|" not in answer
    assert "REFS" not in answer
    assert "TBD" not in answer
    assert "[1]" in answer
    assert "### 작업 절차" in answer
    assert "1." in answer
    assert "EFEM" in answer


def test_setup_ans_v2_runtime_uses_v3_prompt_with_procedure_first_constraints() -> None:
    spec = load_prompt_spec("v2")
    system = spec.setup_ans.system

    assert spec.setup_ans.version == "v3"
    assert "근거 사용 우선순위" in system
    assert "Work Procedure" in system
    assert "## 작업 절차" in system
    assert "이모지 번호" in system
