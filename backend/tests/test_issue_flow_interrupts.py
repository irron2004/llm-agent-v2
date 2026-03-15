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
from backend.llm_infrastructure.llm.langgraph_agent import ISSUE_CASE_EMPTY_MESSAGE, PromptSpec
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent


class SequencedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_model is not None:
            raise NotImplementedError
        if not self._responses:
            return LLMResponse(text="")
        return LLMResponse(text=self._responses.pop(0))


class StubSearchService:
    reranker = None

    def __init__(self, docs: list[RetrievalResult]) -> None:
        self._docs = docs

    def search(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        return list(self._docs)

    def fetch_doc_chunks(self, doc_id: str) -> list[RetrievalResult]:
        return [doc for doc in self._docs if str(doc.doc_id) == str(doc_id)]


def _prompt(name: str) -> PromptTemplate:
    return PromptTemplate(
        name=name, version="v1", system="", user="{sys.query}\n{ref_text}", raw={}
    )


def _make_spec() -> PromptSpec:
    base = _prompt("base")
    return PromptSpec(
        router=PromptTemplate(name="router", version="v1", system="", user="{sys.query}", raw={}),
        setup_mq=PromptTemplate(
            name="setup_mq", version="v1", system="", user="{sys.query}", raw={}
        ),
        ts_mq=PromptTemplate(name="ts_mq", version="v1", system="", user="{sys.query}", raw={}),
        general_mq=PromptTemplate(
            name="general_mq", version="v1", system="", user="{sys.query}", raw={}
        ),
        st_gate=PromptTemplate(name="st_gate", version="v1", system="", user="{sys.query}", raw={}),
        st_mq=PromptTemplate(name="st_mq", version="v1", system="", user="{sys.query}", raw={}),
        setup_ans=base,
        ts_ans=base,
        general_ans=base,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        issue_ans=base,
        issue_detail_ans=base,
    )


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    interrupts = result.get("__interrupt__") or []
    assert interrupts, "expected interrupt payload"
    first = interrupts[0]
    return first.value if hasattr(first, "value") else first


def _issue_docs() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            doc_id="doc-1",
            content="LP1 mapping arm open alarm case",
            score=0.9,
            metadata={"title": "Case 1"},
            raw_text="case 1 summary",
        ),
        RetrievalResult(
            doc_id="doc-2",
            content="RFID read failure case",
            score=0.8,
            metadata={"title": "Case 2"},
            raw_text="case 2 summary",
        ),
    ]


def test_issue_flow_interrupt_ordering_and_loop() -> None:
    agent = LangGraphRAGAgent(
        llm=SequencedLLM(["summary answer", "detail answer"]),
        search_service=StubSearchService(_issue_docs()),
        prompt_spec=_make_spec(),
        mode="verified",
    )

    thread_id = "issue-flow-thread"
    result1 = agent.run(
        "Door Open Alarm",
        thread_id=thread_id,
        state_overrides={"task_mode": "issue", "mq_mode": "off"},
    )
    payload1 = _interrupt_payload(result1)
    assert payload1["type"] == "issue_confirm"
    assert payload1["stage"] == "post_summary"
    assert result1.get("answer") == "summary answer"

    config = {"configurable": {"thread_id": thread_id}}
    result2 = agent._graph.invoke(
        Command(
            resume={
                "type": "issue_confirm",
                "nonce": payload1["nonce"],
                "stage": "post_summary",
                "confirm": True,
            }
        ),
        config,
    )
    payload2 = _interrupt_payload(result2)
    assert payload2["type"] == "issue_case_selection"

    result3 = agent._graph.invoke(
        Command(
            resume={
                "type": "issue_case_selection",
                "nonce": payload2["nonce"],
                "selected_doc_id": "doc-1",
            }
        ),
        config,
    )
    payload3 = _interrupt_payload(result3)
    assert payload3["type"] == "issue_sop_confirm"
    detail_answer = str(result3.get("answer") or "")
    assert "detail answer" in detail_answer
    assert "## 이슈 내용" in detail_answer
    assert "## 해결 방안" in detail_answer

    result4 = agent._graph.invoke(
        Command(
            resume={
                "type": "issue_sop_confirm",
                "nonce": payload3["nonce"],
                "confirm": False,
            }
        ),
        config,
    )
    payload4 = _interrupt_payload(result4)
    assert payload4["type"] == "issue_confirm"
    assert payload4["stage"] == "post_detail"

    result5 = agent._graph.invoke(
        Command(
            resume={
                "type": "issue_confirm",
                "nonce": payload4["nonce"],
                "stage": "post_detail",
                "confirm": True,
            }
        ),
        config,
    )
    payload5 = _interrupt_payload(result5)
    assert payload5["type"] == "issue_case_selection"
    assert [case["doc_id"] for case in payload5["cases"]] == ["doc-1", "doc-2"]


def test_issue_flow_rejects_nonce_mismatch_for_selection() -> None:
    agent = LangGraphRAGAgent(
        llm=SequencedLLM(["summary answer"]),
        search_service=StubSearchService(_issue_docs()),
        prompt_spec=_make_spec(),
        mode="verified",
    )

    thread_id = "issue-flow-bad-nonce"
    first = agent.run(
        "Door Open Alarm",
        thread_id=thread_id,
        state_overrides={"task_mode": "issue", "mq_mode": "off"},
    )
    payload1 = _interrupt_payload(first)

    config = {"configurable": {"thread_id": thread_id}}
    second = agent._graph.invoke(
        Command(
            resume={
                "type": "issue_confirm",
                "nonce": payload1["nonce"],
                "stage": "post_summary",
                "confirm": True,
            }
        ),
        config,
    )
    payload2 = _interrupt_payload(second)
    assert payload2["type"] == "issue_case_selection"

    third = agent._graph.invoke(
        Command(
            resume={
                "type": "issue_case_selection",
                "nonce": "wrong-nonce",
                "selected_doc_id": "doc-1",
            }
        ),
        config,
    )
    assert not third.get("__interrupt__")


def test_issue_flow_no_docs_returns_graceful_answer_without_interrupt() -> None:
    agent = LangGraphRAGAgent(
        llm=SequencedLLM(["unused-summary"]),
        search_service=StubSearchService([]),
        prompt_spec=_make_spec(),
        mode="verified",
    )

    result = agent.run(
        "No matching issue",
        thread_id="issue-empty-thread",
        state_overrides={"task_mode": "issue", "mq_mode": "off"},
    )
    assert not result.get("__interrupt__")
    assert result.get("answer") == ISSUE_CASE_EMPTY_MESSAGE


def test_device_selection_issue_doc_types_sets_task_mode_and_route(
    monkeypatch: Any,
) -> None:
    def _fake_interrupt(_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "device_selection",
            "selected_devices": ["SUPRA N"],
            "selected_doc_types": ["myservice", "gcb", "ts"],
        }

    monkeypatch.setattr(langgraph_agent_module, "interrupt", _fake_interrupt)

    state: dict[str, Any] = {
        "query": "door open alarm",
        "route": "setup",
        "mq_mode": "off",
    }

    command = langgraph_agent_module.device_selection_node(
        state,
        device_fetcher=lambda: {
            "devices": [{"name": "SUPRA N", "doc_count": 1}],
            "doc_types": [
                {"name": "myservice", "doc_count": 1},
                {"name": "gcb", "doc_count": 1},
                {"name": "ts", "doc_count": 1},
            ],
        },
    )

    assert isinstance(command, Command)
    assert command.goto == "prepare_retrieve"
    update = command.update or {}
    assert update.get("task_mode") == "issue"
    assert update.get("route") == "general"
    parsed_query = update.get("parsed_query") or {}
    assert parsed_query.get("task_mode") == "issue"
    assert parsed_query.get("route") == "general"
