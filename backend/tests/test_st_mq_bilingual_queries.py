from __future__ import annotations

from typing import Iterable

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec, st_mq_node
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate


class DummyLLM(BaseLLM):
    def generate(self, messages: Iterable[dict[str, str]], *, response_model=None, **kwargs):
        if response_model is not None:
            raise NotImplementedError
        message_list = list(messages)
        user_text = message_list[-1]["content"] if message_list else ""

        if "|" in user_text:
            query, target = user_text.split("|", 1)
            query = query.strip()
            target = target.strip()
            if target == "Korean":
                return LLMResponse(text=f"한국어 {query}")
            if target == "English":
                return LLMResponse(text=f"english {query}")
        return LLMResponse(text='{"queries":["query a","query b","query c"]}')


def _prompt(name: str, user: str = "") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system="", user=user, raw={})


def _make_spec() -> PromptSpec:
    empty = _prompt("empty", user="")
    return PromptSpec(
        router=empty,
        setup_mq=empty,
        ts_mq=empty,
        general_mq=empty,
        st_gate=empty,
        st_mq=_prompt("st_mq", user="{sys.query}"),
        setup_ans=empty,
        ts_ans=empty,
        general_ans=empty,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        translate=_prompt("translate", user="{query}|{target_language}"),
    )


def _contains_korean(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def test_st_mq_node_skip_mq_keeps_final_bilingual_six_queries() -> None:
    state = {
        "query": "fcip source 교체 후 power cal하는 방법",
        "query_en": "how to run power calibration after fcip source replacement",
        "query_ko": "fcip source 교체 후 power cal 절차",
        "route": "setup",
        "skip_mq": True,
        "search_queries": [
            "fcip source replacement power calibration",
            "power cal setup parameters fcip source",
            "fcip source replacement acceptance criteria",
            "fcip 소스 교체 power cal 절차",
            "fcip 소스 교체 설정값",
            "fcip 소스 교체 허용오차 기준",
        ],
    }

    result = st_mq_node(state, llm=DummyLLM(), spec=_make_spec())
    queries = result["search_queries"]

    assert len(queries) == 6
    assert all(not _contains_korean(q) for q in queries[:3])
    assert all(_contains_korean(q) for q in queries[3:])


def test_st_mq_node_skip_mq_translates_to_fill_korean_queries() -> None:
    state = {
        "query": "how to increase ashing rate",
        "query_en": "how to increase ashing rate",
        "query_ko": "어싱 레이트를 높이는 방법",
        "route": "setup",
        "skip_mq": True,
        "search_queries": [
            "increase ashing rate setup",
            "ashing rate low troubleshooting",
            "ashing rate parameter adjustment",
        ],
    }

    result = st_mq_node(state, llm=DummyLLM(), spec=_make_spec())
    queries = result["search_queries"]

    assert len(queries) == 6
    assert all(not _contains_korean(q) for q in queries[:3])
    assert all(_contains_korean(q) for q in queries[3:])
