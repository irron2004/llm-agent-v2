from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm import langgraph_agent as langgraph_agent_module
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    _infer_task_mode_from_doc_types,
    answer_node,
    issue_detail_answer_node,
    mq_node,
    route_node,
)
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate


class SequencedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.system_prompts: list[str] = []

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_model is not None:
            raise NotImplementedError
        msgs = list(messages)
        for message in msgs:
            if message.get("role") == "system":
                self.system_prompts.append(str(message.get("content") or ""))
        if not self._responses:
            return LLMResponse(text="")
        return LLMResponse(text=self._responses.pop(0))


def _prompt(name: str, system: str = "", user: str = "{sys.query}\n{ref_text}") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system=system, user=user, raw={})


def _make_spec(*, issue_mq_system: str = "") -> PromptSpec:
    base = _prompt("base")
    return PromptSpec(
        router=_prompt("router", user="{sys.query}"),
        setup_mq=_prompt("setup_mq", system="SETUP_MQ_SYSTEM", user="{sys.query}"),
        ts_mq=_prompt("ts_mq", system="TS_MQ_SYSTEM", user="{sys.query}"),
        general_mq=_prompt("general_mq", system="GENERAL_MQ_SYSTEM", user="{sys.query}"),
        st_gate=_prompt("st_gate", user="{sys.query}"),
        st_mq=_prompt("st_mq", user="{sys.query}"),
        setup_ans=base,
        ts_ans=base,
        general_ans=base,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        issue_ans=base,
        issue_detail_ans=base,
        issue_mq=(
            _prompt("issue_mq", system=issue_mq_system, user="{sys.query}")
            if issue_mq_system
            else None
        ),
    )


def _make_refs(count: int) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        refs.append(
            {
                "rank": idx,
                "doc_id": f"doc-{idx}",
                "content": f"content-{idx}",
                "metadata": {"doc_type": "myservice", "section": "action"},
            }
        )
    return refs


def test_infer_task_mode_ts_only() -> None:
    assert _infer_task_mode_from_doc_types(["ts"]) == "ts"
    assert _infer_task_mode_from_doc_types(["Trouble Shooting Guide"]) == "ts"
    assert _infer_task_mode_from_doc_types(["myservice", "ts"]) == "issue"


def test_route_node_forces_ts_when_task_mode_is_ts() -> None:
    state: dict[str, Any] = {"query": "alarm", "task_mode": "ts", "parsed_query": {}}
    result = route_node(state, llm=SequencedLLM(["unused"]), spec=_make_spec())
    assert result["route"] == "ts"
    assert result["task_mode"] == "ts"


def test_mq_node_uses_issue_mq_template_when_issue_mode() -> None:
    llm = SequencedLLM(
        [
            "door open alarm\nroot cause check\nrecovery procedure",
            "도어 오픈 알람\n원인 점검\n복구 절차",
        ]
    )
    state: dict[str, Any] = {
        "route": "general",
        "task_mode": "issue",
        "query": "door open alarm",
        "query_en": "door open alarm",
        "query_ko": "도어 오픈 알람",
    }
    result = mq_node(state, llm=llm, spec=_make_spec(issue_mq_system="ISSUE_MQ_SYSTEM"))
    assert result["general_mq_list"]
    assert result["setup_mq_list"] == []
    assert result["ts_mq_list"] == []
    assert llm.system_prompts
    assert all("ISSUE_MQ_SYSTEM" in text for text in llm.system_prompts)


def test_answer_node_issue_splits_case_refs_and_answer_refs() -> None:
    llm = SequencedLLM(["summary answer"])
    state: dict[str, Any] = {
        "route": "general",
        "task_mode": "issue",
        "query": "door open alarm",
        "answer_ref_json": _make_refs(6),
    }
    result = answer_node(state, llm=llm, spec=_make_spec())
    assert len(result["answer_ref_json"]) == 5
    assert len(result["issue_top10_cases"]) == 6
    assert len(result["issue_case_refs"]) == 6
    assert len(result["issue_case_ref_map"]) == 6


def test_issue_detail_answer_node_prefers_issue_case_ref_map() -> None:
    llm = SequencedLLM(["detail answer"])
    state: dict[str, Any] = {
        "route": "general",
        "task_mode": "issue",
        "query": "door open alarm",
        "issue_selected_doc_id": "doc-2",
        "answer_ref_json": _make_refs(1),
        "issue_case_ref_map": {
            "doc-2": [
                {
                    "rank": 1,
                    "doc_id": "doc-2",
                    "content": "mapped detail",
                    "metadata": {"doc_type": "gcb", "section": "detail"},
                }
            ]
        },
    }
    result = issue_detail_answer_node(state, llm=llm, spec=_make_spec())
    assert result["answer_ref_json"]
    assert result["answer_ref_json"][0]["doc_id"] == "doc-2"


def test_device_selection_ts_only_sets_ts_route(monkeypatch: Any) -> None:
    def _fake_interrupt(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "device_selection",
            "selected_devices": ["SUPRA N"],
            "selected_doc_types": ["ts"],
        }

    monkeypatch.setattr(langgraph_agent_module, "interrupt", _fake_interrupt)

    state: dict[str, Any] = {
        "query": "alarm",
        "route": "general",
        "mq_mode": "off",
    }
    command = langgraph_agent_module.device_selection_node(
        state,
        device_fetcher=lambda: {
            "devices": [{"name": "SUPRA N", "doc_count": 1}],
            "doc_types": [{"name": "ts", "doc_count": 1}],
        },
    )
    assert isinstance(command, Command)
    assert command.goto == "prepare_retrieve"
    update = command.update or {}
    assert update.get("task_mode") == "ts"
    assert update.get("route") == "ts"
    parsed_query = update.get("parsed_query") or {}
    assert parsed_query.get("task_mode") == "ts"
    assert parsed_query.get("route") == "ts"
