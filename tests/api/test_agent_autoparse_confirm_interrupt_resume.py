from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any, TypeVar, cast, overload
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from backend.api import dependencies
from backend.api.routers import agent as agent_router
from backend.domain.doc_type_mapping import expand_doc_type_selection
from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent


TModel = TypeVar("TModel", bound=BaseModel)


class _StubLLM(BaseLLM):
    @overload
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[TModel],
        **kwargs: object,
    ) -> TModel: ...

    @overload
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: None = None,
        **kwargs: object,
    ) -> LLMResponse: ...

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[TModel] | None = None,
        **_: Any,
    ) -> LLMResponse | TModel:
        msg_list = list(messages)
        system = msg_list[0].get("content", "") if msg_list else ""

        if "router-system" in system:
            text = "general"
        elif "st-gate-system" in system:
            text = "no_st"
        elif "judge-system" in system:
            text = '{"faithful": true, "issues": [], "hint": ""}'
        elif "answer-system" in system:
            text = "answer"
        elif (
            "setup-mq-system" in system
            or "ts-mq-system" in system
            or "general-mq-system" in system
            or "st-mq-system" in system
        ):
            text = '["mq synthetic"]'
        else:
            text = "general"

        if response_model is None:
            return LLMResponse(text=text)
        return response_model.model_validate({})


class _FakeSearchService:
    reranker = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(
        self, query: str, top_k: int = 10, **kwargs: Any
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "kwargs": dict(kwargs)})

        doc_types_raw = kwargs.get("doc_types") or ["manual"]
        if not isinstance(doc_types_raw, list):
            doc_types = [str(doc_types_raw)]
        else:
            doc_types = [str(v) for v in doc_types_raw] or ["manual"]

        docs: list[RetrievalResult] = []
        for idx, doc_type in enumerate(doc_types[: max(top_k, 1)]):
            docs.append(
                RetrievalResult(
                    doc_id=f"doc-{idx + 1}",
                    content=f"{query}-{doc_type}",
                    score=max(0.2, 0.98 - (idx * 0.05)),
                    metadata={
                        "title": f"Doc {idx + 1}",
                        "doc_type": doc_type,
                        "device_name": "SUPRA N",
                        "equip_id": "EPAG50",
                        "page": idx + 1,
                    },
                    raw_text=f"{query}-{doc_type}",
                )
            )
        return docs[:top_k]


def _prompt_spec() -> SimpleNamespace:
    return SimpleNamespace(
        router=SimpleNamespace(system="router-system", user="{sys.query}"),
        setup_mq=SimpleNamespace(system="setup-mq-system", user="{sys.query}"),
        ts_mq=SimpleNamespace(system="ts-mq-system", user="{sys.query}"),
        general_mq=SimpleNamespace(system="general-mq-system", user="{sys.query}"),
        st_gate=SimpleNamespace(
            system="st-gate-system", user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}"
        ),
        st_mq=SimpleNamespace(
            system="st-mq-system",
            user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}|{st_gate}",
        ),
        setup_ans=SimpleNamespace(
            system="answer-system", user="{sys.query}|{ref_text}"
        ),
        ts_ans=SimpleNamespace(system="answer-system", user="{sys.query}|{ref_text}"),
        general_ans=SimpleNamespace(
            system="answer-system", user="{sys.query}|{ref_text}"
        ),
        judge_setup_sys="judge-system",
        judge_ts_sys="judge-system",
        judge_general_sys="judge-system",
        auto_parse=SimpleNamespace(system="auto-parse-system", user="{query}"),
        translate=None,
        setup_ans_en=None,
        setup_ans_zh=None,
        setup_ans_ja=None,
        ts_ans_en=None,
        ts_ans_zh=None,
        ts_ans_ja=None,
        general_ans_en=None,
        general_ans_zh=None,
        general_ans_ja=None,
    )


@pytest.fixture
def _real_flow_overrides(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_FakeSearchService]:
    def _safe_wrap_node(self: LangGraphRAGAgent, _name: str, fn: Any) -> Any:
        def _wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
            result = fn(state, *args, **kwargs)
            if isinstance(result, dict):
                result.pop("_events", None)
            return result

        return _wrapped

    fake_search = _FakeSearchService()
    app = cast(FastAPI, client.app)

    app.dependency_overrides[dependencies.get_default_llm] = lambda: _StubLLM()
    app.dependency_overrides[dependencies.get_search_service] = lambda: fake_search
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = _prompt_spec

    monkeypatch.setattr(agent_router, "_checkpointer", MemorySaver())
    monkeypatch.setattr(LangGraphRAGAgent, "_wrap_node", _safe_wrap_node)

    try:
        yield fake_search
    finally:
        app.dependency_overrides.pop(dependencies.get_default_llm, None)
        app.dependency_overrides.pop(dependencies.get_search_service, None)
        app.dependency_overrides.pop(dependencies.get_prompt_spec_cached, None)


def _run(
    client: TestClient,
    *,
    payload: dict[str, Any],
    expected_status: int = 200,
) -> dict[str, Any]:
    response = client.post(
        "/api/agent/run",
        json={
            "message": "SUPRA N issue checklist",
            "auto_parse": True,
            "max_attempts": 0,
            "top_k": 3,
            "mq_mode": "off",
            **payload,
        },
    )
    assert response.status_code == expected_status
    return cast(dict[str, Any], response.json())


def test_interrupt_gate(
    client: TestClient, _real_flow_overrides: _FakeSearchService
) -> None:
    data = _run(
        client,
        payload={
            "guided_confirm": False,
            "thread_id": f"interrupt-gate-{uuid4().hex}",
        },
    )
    assert data["interrupted"] is False
    assert data["interrupt_payload"] is None


def test_interrupt_when_enabled(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    data = _run(
        client,
        payload={
            "guided_confirm": True,
            "thread_id": f"interrupt-on-{uuid4().hex}",
        },
    )
    assert data["interrupted"] is True
    interrupt_payload = cast(dict[str, Any], data["interrupt_payload"])
    assert interrupt_payload["type"] == "auto_parse_confirm"


def test_resume_applies_task_mode_scope(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    sop_first = _run(
        client,
        payload={
            "guided_confirm": True,
            "thread_id": f"sop-{uuid4().hex}",
        },
    )
    assert sop_first["interrupted"] is True

    _real_flow_overrides.calls.clear()
    sop_resume = _run(
        client,
        payload={
            "thread_id": sop_first["thread_id"],
            "resume_decision": {
                "type": "auto_parse_confirm",
                "target_language": "ko",
                "selected_device": "__skip__",
                "selected_equip_id": "__skip__",
                "task_mode": "sop",
            },
        },
    )
    assert sop_resume["interrupted"] is False
    sop_expected = set(expand_doc_type_selection(["sop"]))
    issue_expected = set(expand_doc_type_selection(["myservice", "gcb", "ts"]))
    sop_calls = [
        call
        for call in _real_flow_overrides.calls
        if (
            (doc_types := set(cast(list[str], call["kwargs"].get("doc_types") or [])))
            and doc_types.issubset(sop_expected)
            and doc_types.isdisjoint(issue_expected)
        )
    ]
    assert sop_calls

    issue_first = _run(
        client,
        payload={
            "guided_confirm": True,
            "thread_id": f"issue-{uuid4().hex}",
        },
    )
    assert issue_first["interrupted"] is True

    _real_flow_overrides.calls.clear()
    issue_resume = _run(
        client,
        payload={
            "thread_id": issue_first["thread_id"],
            "resume_decision": {
                "type": "auto_parse_confirm",
                "target_language": "ko",
                "selected_device": "__skip__",
                "selected_equip_id": "__skip__",
                "task_mode": "issue",
            },
        },
    )
    assert issue_resume["interrupted"] is True
    interrupt_payload = cast(dict[str, Any], issue_resume["interrupt_payload"])
    assert interrupt_payload["type"] == "issue_confirm"
    issue_calls = [
        call
        for call in _real_flow_overrides.calls
        if (
            (doc_types := set(cast(list[str], call["kwargs"].get("doc_types") or [])))
            and doc_types.issubset(issue_expected)
            and doc_types.isdisjoint(sop_expected)
        )
    ]
    assert issue_calls


def test_target_language_metadata(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    first = _run(
        client,
        payload={
            "guided_confirm": True,
            "thread_id": f"lang-{uuid4().hex}",
        },
    )
    assert first["interrupted"] is True

    resumed = _run(
        client,
        payload={
            "thread_id": first["thread_id"],
            "resume_decision": {
                "type": "auto_parse_confirm",
                "target_language": "en",
                "selected_device": "__skip__",
                "selected_equip_id": "__skip__",
                "task_mode": "issue",
            },
        },
    )
    metadata = cast(dict[str, Any], resumed["metadata"])
    assert metadata["target_language"] == "en"


def test_filter_doc_types_issue_scope_forces_issue_mode(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    _real_flow_overrides.calls.clear()
    result = _run(
        client,
        payload={
            "guided_confirm": False,
            "filter_doc_types": ["myservice", "gcb", "ts"],
        },
    )

    assert result["interrupted"] is True
    interrupt_payload = cast(dict[str, Any], result["interrupt_payload"])
    assert interrupt_payload["type"] == "issue_confirm"

    metadata = cast(dict[str, Any], result["metadata"])
    assert metadata.get("selected_task_mode") == "issue"
    assert metadata.get("route") == "general"

    issue_expected = set(expand_doc_type_selection(["myservice", "gcb", "ts"]))
    issue_calls = [
        call
        for call in _real_flow_overrides.calls
        if (
            (doc_types := set(cast(list[str], call["kwargs"].get("doc_types") or [])))
            and doc_types.issubset(issue_expected)
        )
    ]
    assert issue_calls


def test_filter_doc_types_issue_scope_interrupt_is_resumable(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    first = _run(
        client,
        payload={
            "guided_confirm": False,
            "thread_id": f"issue-filter-{uuid4().hex}",
            "filter_doc_types": ["myservice", "gcb", "ts"],
        },
    )
    assert first["interrupted"] is True
    first_payload = cast(dict[str, Any], first["interrupt_payload"])
    assert first_payload["type"] == "issue_confirm"

    resumed = _run(
        client,
        payload={
            "thread_id": cast(str, first["thread_id"]),
            "resume_decision": {
                "type": "issue_confirm",
                "nonce": first_payload["nonce"],
                "stage": "post_summary",
                "confirm": False,
            },
        },
    )

    assert resumed["interrupted"] is False


def test_followup_query_can_narrow_strict_issue_scope_to_gcb_only(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    thread_id = f"issue-to-gcb-{uuid4().hex}"

    first = _run(
        client,
        payload={
            "guided_confirm": False,
            "thread_id": thread_id,
            "filter_doc_types": ["myservice", "gcb", "ts"],
        },
    )
    assert first["interrupted"] is True

    _real_flow_overrides.calls.clear()

    second = _run(
        client,
        payload={
            "thread_id": thread_id,
            "message": "gcb 문서만 보여줘",
            "guided_confirm": False,
        },
    )
    assert second["interrupted"] in {True, False}

    gcb_expected = set(expand_doc_type_selection(["gcb"]))
    issue_all = set(expand_doc_type_selection(["myservice", "gcb", "ts"]))
    non_gcb_issue = issue_all - gcb_expected

    gcb_only_calls = [
        call
        for call in _real_flow_overrides.calls
        if (
            (doc_types := set(cast(list[str], call["kwargs"].get("doc_types") or [])))
            and doc_types.issubset(gcb_expected)
            and doc_types.isdisjoint(non_gcb_issue)
        )
    ]
    assert gcb_only_calls


def test_missing_checkpoint_400(
    client: TestClient,
    _real_flow_overrides: _FakeSearchService,
) -> None:
    data = _run(
        client,
        expected_status=400,
        payload={
            "guided_confirm": True,
            "thread_id": f"missing-{uuid4().hex}",
            "resume_decision": {"type": "auto_parse_confirm", "task_mode": "issue"},
        },
    )
    assert "No checkpoint for thread_id" in cast(str, data["detail"])
