from __future__ import annotations

from typing import Iterable

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec, answer_node, load_prompt_spec
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate


class CaptureSystemLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, messages: Iterable[dict[str, str]], *, response_model=None, **kwargs):
        if response_model is not None:
            raise NotImplementedError
        message_list = list(messages)
        if message_list and message_list[0].get("role") == "system":
            self.last_system = message_list[0].get("content", "")
        if message_list:
            self.last_user = message_list[-1].get("content", "")
        return LLMResponse(text="ok")


def _prompt(name: str, system: str = "", user: str = "{sys.query}\n{ref_text}") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system=system, user=user, raw={})


def _spec_with_zh_prompts() -> PromptSpec:
    empty = _prompt("empty")
    return PromptSpec(
        router=empty,
        setup_mq=empty,
        ts_mq=empty,
        general_mq=empty,
        st_gate=empty,
        st_mq=empty,
        setup_ans=_prompt("setup_ans", system="KO_SETUP"),
        ts_ans=_prompt("ts_ans", system="KO_TS"),
        general_ans=_prompt("general_ans", system="KO_GENERAL"),
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        setup_ans_en=_prompt("setup_ans_en", system="EN_SETUP"),
        ts_ans_en=_prompt("ts_ans_en", system="EN_TS"),
        general_ans_en=_prompt("general_ans_en", system="EN_GENERAL"),
        setup_ans_zh=_prompt("setup_ans_zh", system="ZH_SETUP"),
        ts_ans_zh=_prompt("ts_ans_zh", system="ZH_TS"),
        general_ans_zh=_prompt("general_ans_zh", system="ZH_GENERAL"),
        setup_ans_ja=_prompt("setup_ans_ja", system="JA_SETUP"),
        ts_ans_ja=_prompt("ts_ans_ja", system="JA_TS"),
        general_ans_ja=_prompt("general_ans_ja", system="JA_GENERAL"),
    )


def test_answer_node_uses_chinese_template_when_detected_language_is_zh() -> None:
    llm = CaptureSystemLLM()
    spec = _spec_with_zh_prompts()
    state = {
        "route": "general",
        "query": "设备校准步骤是什么？",
        "query_en": "what is the calibration procedure?",
        "detected_language": "zh",
        "ref_json": [],
    }

    result = answer_node(state, llm=llm, spec=spec)

    assert result["answer"] == "ok"
    assert llm.last_system == "ZH_GENERAL"
    assert llm.last_user is not None
    assert "设备校准步骤是什么？" in llm.last_user


def test_load_prompt_spec_includes_chinese_templates() -> None:
    spec = load_prompt_spec("v1")

    assert spec.setup_ans_zh is not None
    assert spec.ts_ans_zh is not None
    assert spec.general_ans_zh is not None
