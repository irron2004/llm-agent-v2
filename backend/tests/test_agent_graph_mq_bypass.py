from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent


class DummyLLM(BaseLLM):
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_model is not None:
            raise NotImplementedError
        message_list = list(messages)
        user_text = message_list[-1]["content"] if message_list else ""
        if "JSON 한 줄로 반환" in user_text:
            return LLMResponse(text='{"faithful": true, "issues": [], "hint": ""}')
        return LLMResponse(text="ok")


class StubSearchService:
    reranker = None

    def search(self, query: str, **kwargs: Any) -> list[Any]:
        return []


def _prompt(name: str, user: str = "") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system="", user=user, raw={})


def _make_spec() -> PromptSpec:
    empty = _prompt("empty", user="{sys.query}")
    return PromptSpec(
        router=empty,
        setup_mq=empty,
        ts_mq=empty,
        general_mq=empty,
        st_gate=empty,
        st_mq=empty,
        setup_ans=_prompt("setup_ans", user="{sys.query}\n{ref_text}"),
        ts_ans=_prompt("ts_ans", user="{sys.query}\n{ref_text}"),
        general_ans=_prompt("general_ans", user="{sys.query}\n{ref_text}"),
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
    )


def test_graph_bypasses_mq_path_on_first_attempt_in_fallback_mode(monkeypatch) -> None:
    def _fail_if_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("MQ path node should not run on fallback attempt 0")

    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.route_node",
        lambda state, **kwargs: {"route": "general"},
    )
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.mq_node", _fail_if_called)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.st_gate_node", _fail_if_called)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.st_mq_node", _fail_if_called)

    agent = LangGraphRAGAgent(
        llm=DummyLLM(),
        search_service=StubSearchService(),
        prompt_spec=_make_spec(),
        mode="base",
    )

    result = agent.run("  chamber pressure alarm  ", state_overrides={"mq_mode": "fallback"})

    assert result["search_queries"] == ["chamber pressure alarm"]
    assert result["mq_used"] is False
    assert result["mq_reason"] == "empty_retrieval"


def test_graph_keeps_existing_mq_path_when_mode_is_on(monkeypatch) -> None:
    markers = {"mq": False, "st_gate": False, "st_mq": False}

    def _route_node(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"route": "general"}

    def _mq_node(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        markers["mq"] = True
        return {"general_mq_list": ["mq generated query"]}

    def _st_gate_node(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        markers["st_gate"] = True
        return {"st_gate": "no_st"}

    def _st_mq_node(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        markers["st_mq"] = True
        return {"search_queries": ["mq generated query"]}

    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.route_node", _route_node)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.mq_node", _mq_node)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.st_gate_node", _st_gate_node)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.st_mq_node", _st_mq_node)

    agent = LangGraphRAGAgent(
        llm=DummyLLM(),
        search_service=StubSearchService(),
        prompt_spec=_make_spec(),
        mode="base",
    )

    agent.run("plasma strike issue", state_overrides={"mq_mode": "on"})

    assert markers == {"mq": True, "st_gate": True, "st_mq": True}
