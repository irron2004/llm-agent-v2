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
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent


class DummyLLM(BaseLLM):
    def __init__(self, *, judge_faithful: bool = False) -> None:
        self.judge_faithful = judge_faithful

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
            faithful = "true" if self.judge_faithful else "false"
            return LLMResponse(
                text=f'{{"faithful": {faithful}, "issues": ["test"], "hint": "retry"}}'
            )
        return LLMResponse(text="ok")


class StubSearchService:
    reranker = None

    def __init__(self, *, docs_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.docs_by_query = docs_by_query

    def search(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        return list(self.docs_by_query.get(query, []))


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


def _doc(doc_id: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        content="snippet",
        score=1.0,
        metadata={"chunk_id": f"chunk::{doc_id}", "doc_type": "setup"},
        raw_text="snippet",
    )


def test_fallback_mode_uses_mq_only_after_empty_retrieval(monkeypatch) -> None:
    markers = {"mq_calls": 0}

    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.route_node",
        lambda state, **kwargs: {"route": "general"},
    )

    def _mq_node(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        markers["mq_calls"] += 1
        return {"general_mq_list": ["mq-query"], "mq_used": True}

    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.mq_node", _mq_node)
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.st_gate_node",
        lambda state, **kwargs: {"st_gate": "no_st"},
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.st_mq_node",
        lambda state, **kwargs: {"search_queries": ["mq-query"]},
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.answer_node",
        lambda state, **kwargs: {"answer": "ok", "reasoning": None},
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.judge_node",
        lambda state, **kwargs: {
            "judge": {
                "faithful": bool(state.get("docs")),
                "issues": [] if state.get("docs") else ["no_docs"],
                "hint": "retry",
            }
        },
    )

    agent = LangGraphRAGAgent(
        llm=DummyLLM(),
        search_service=StubSearchService(
            docs_by_query={
                "mq-query": [_doc("doc::mq")],
            }
        ),
        prompt_spec=_make_spec(),
        mode="verified",
    )

    result = agent.run(
        "chamber pressure alarm",
        max_attempts=3,
        state_overrides={"mq_mode": "fallback"},
    )

    assert markers["mq_calls"] == 1
    assert result["mq_used"] is True
    assert result["mq_reason"] == "empty_retrieval"
    assert result["attempts"] == 1


def test_mq_mode_off_never_uses_mq_and_stops_at_hard_ceiling(monkeypatch) -> None:
    def _fail_if_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("MQ path node should never run in mq_mode=off")

    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.route_node",
        lambda state, **kwargs: {"route": "general"},
    )
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.mq_node", _fail_if_called)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.st_gate_node", _fail_if_called)
    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.st_mq_node", _fail_if_called)
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.refine_queries_node",
        lambda state, **kwargs: {"search_queries": ["stable query"]},
    )

    agent = LangGraphRAGAgent(
        llm=DummyLLM(judge_faithful=False),
        search_service=StubSearchService(docs_by_query={"stable query": [_doc("doc::stable")]}),
        prompt_spec=_make_spec(),
        mode="verified",
    )

    result = agent.run(
        "stable query",
        max_attempts=2,
        state_overrides={"mq_mode": "off"},
    )

    assert result.get("mq_used") is False
    assert result.get("mq_reason") is None
    assert result.get("attempts") == 2
    assert result["judge"]["faithful"] is False
    assert "max_attempts" in result["judge"].get("hint", "")


def test_fallback_mode_sets_unfaithful_reason_after_attempt_two(monkeypatch) -> None:
    markers = {"mq_calls": 0}

    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.route_node",
        lambda state, **kwargs: {"route": "general"},
    )

    def _mq_node(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        markers["mq_calls"] += 1
        return {"general_mq_list": ["mq-query"], "mq_used": True}

    monkeypatch.setattr("backend.services.agents.langgraph_rag_agent.mq_node", _mq_node)
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.st_gate_node",
        lambda state, **kwargs: {"st_gate": "no_st"},
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.st_mq_node",
        lambda state, **kwargs: {"search_queries": ["mq-query"]},
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.refine_queries_node",
        lambda state, **kwargs: {"search_queries": ["stable query"]},
    )

    agent = LangGraphRAGAgent(
        llm=DummyLLM(judge_faithful=False),
        search_service=StubSearchService(
            docs_by_query={
                "stable query": [_doc("doc::stable")],
                "mq-query": [_doc("doc::mq")],
            }
        ),
        prompt_spec=_make_spec(),
        mode="verified",
    )

    result = agent.run(
        "stable query",
        max_attempts=3,
        state_overrides={"mq_mode": "fallback"},
    )

    assert markers["mq_calls"] == 1
    assert result.get("mq_used") is True
    assert result.get("attempts") == 3
    assert result.get("mq_reason") == "unfaithful_after_deterministic_retries"
    assert result["judge"]["faithful"] is False
    assert "max_attempts" in result["judge"].get("hint", "")
