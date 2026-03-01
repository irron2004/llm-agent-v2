from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    TEMP_QUERY_GEN,
    mq_node,
    refine_queries_node,
    resolve_querygen_temperature,
    st_mq_node,
)
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate


class RecordingLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_model is not None:
            raise NotImplementedError
        self.calls.append(dict(kwargs))
        if not self._responses:
            raise AssertionError("No LLM response prepared for call")
        return LLMResponse(text=self._responses.pop(0))


def _prompt(name: str, *, system: str = "", user: str = "") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system=system, user=user, raw={})


def _make_spec() -> PromptSpec:
    empty = _prompt("empty")
    return PromptSpec(
        router=empty,
        setup_mq=_prompt("setup_mq", system="setup-sys", user="{sys.query}"),
        ts_mq=_prompt("ts_mq", system="ts-sys", user="{sys.query}"),
        general_mq=_prompt("general_mq", system="general-sys", user="{sys.query}"),
        st_gate=empty,
        st_mq=_prompt(
            "st_mq",
            system="st-mq-sys",
            user="{sys.query}\n{setup_mq}\n{ts_mq}\n{general_mq}\n{st_gate}",
        ),
        setup_ans=empty,
        ts_ans=empty,
        general_ans=empty,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        translate=None,
    )


@pytest.mark.parametrize(
    "mq_mode,attempts,mq_invoked,expected",
    [
        ("on", 0, False, TEMP_QUERY_GEN),
        ("off", 0, False, 0.0),
        ("fallback", 1, False, 0.0),
        ("fallback", 2, True, 0.0),
        ("fallback", 2, False, TEMP_QUERY_GEN),
    ],
)
def test_resolve_querygen_temperature_policy(
    mq_mode: str,
    attempts: int,
    mq_invoked: bool,
    expected: float,
) -> None:
    state = {"mq_mode": mq_mode, "attempts": attempts}
    assert resolve_querygen_temperature(state, mq_invoked=mq_invoked) == expected


def test_mq_node_uses_deterministic_temperature_in_fallback_when_invoked() -> None:
    llm = RecordingLLM(
        responses=[
            '{"queries": ["pump failure alarm", "pump alarm cause", "pump alarm troubleshoot"]}',
            '{"queries": ["펌프 알람", "펌프 알람 원인", "펌프 알람 조치"]}',
        ]
    )
    state = {
        "route": "general",
        "query": "pump failure alarm",
        "query_en": "pump failure alarm",
        "query_ko": "펌프 고장 알람",
        "mq_mode": "fallback",
        "attempts": 2,
    }

    mq_node(state, llm=llm, spec=_make_spec())

    assert [call.get("temperature") for call in llm.calls] == [0.0, 0.0]


def test_st_mq_node_keeps_temp_query_gen_in_on_mode() -> None:
    llm = RecordingLLM(
        responses=[
            '{"queries": ["pump failure alarm", "pump alarm cause", "pump alarm troubleshoot", "펌프 고장 알람", "펌프 알람 원인", "펌프 알람 점검"]}'
        ]
    )
    state = {
        "query": "pump failure alarm",
        "query_en": "pump failure alarm",
        "query_ko": "펌프 고장 알람",
        "route": "general",
        "setup_mq_list": [],
        "ts_mq_list": [],
        "general_mq_list": ["pump failure alarm"],
        "general_mq_ko_list": ["펌프 고장 알람", "펌프 알람 원인", "펌프 알람 점검"],
        "st_gate": "no_st",
        "mq_mode": "on",
        "attempts": 0,
    }

    st_mq_node(state, llm=llm, spec=_make_spec())

    assert [call.get("temperature") for call in llm.calls] == [TEMP_QUERY_GEN]


def test_refine_queries_node_uses_zero_temp_in_off_mode_early_attempts() -> None:
    llm = RecordingLLM(
        responses=[
            '{"queries": ["pump failure alarm", "pump alarm root cause", "pump alarm recovery"]}'
        ]
    )
    state = {
        "query": "pump failure alarm",
        "query_en": "pump failure alarm",
        "query_ko": "펌프 고장 알람",
        "route": "general",
        "search_queries": ["pump alarm"],
        "general_mq_ko_list": ["펌프 고장 알람", "펌프 알람 원인", "펌프 알람 조치"],
        "judge": {"hint": "add root cause angle"},
        "mq_mode": "off",
        "attempts": 1,
    }

    refine_queries_node(state, llm=llm, spec=_make_spec())

    assert [call.get("temperature") for call in llm.calls] == [0.0]
