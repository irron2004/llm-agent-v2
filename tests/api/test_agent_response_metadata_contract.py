from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import agent as agent_router


class _MinimalSearchService:
    def __init__(self) -> None:
        self.es_engine = SimpleNamespace(index_name="pe_docs_dev_current")

    def search(self, query: str, top_k: int = 10, **_: object) -> list[Any]:
        del query, top_k
        return []


class _FakeAutoParseAgent:
    def __init__(self, event_sink: Any = None) -> None:
        self._event_sink = event_sink

    def run(self, *_: Any, **__: Any) -> dict[str, Any]:
        if callable(self._event_sink):
            self._event_sink({"type": "node_end", "node": "retrieve"})
        return {
            "answer": "ok",
            "judge": {"faithful": True, "issues": [], "hint": ""},
            "display_docs": [],
            "docs": [],
            "route": "prepare_retrieve",
            "st_gate": "strict",
            "mq_used": False,
            "mq_reason": None,
            "attempts": 1,
            "retry_strategy": "refine_queries",
            "guardrail_dropped_numeric": 2,
            "guardrail_dropped_anchor": 1,
            "guardrail_final_count": 2,
            "search_queries": ["pump alarm reset", "pump alarm procedure"],
            "general_mq_list": [
                "",
                "x" * 180,
                "pump alarm procedure with 123",
                "pump alarm procedure with 123",
                "  ",
                "query-5",
                "query-6",
                "query-7",
            ],
        }


class _FakeNonIssueAnswerRefsAgent:
    def __init__(self, event_sink: Any = None) -> None:
        self._event_sink = event_sink

    def run(self, *_: Any, **__: Any) -> dict[str, Any]:
        if callable(self._event_sink):
            self._event_sink({"type": "node_end", "node": "retrieve"})
        return {
            "answer": "ok",
            "judge": {"faithful": True, "issues": [], "hint": ""},
            "display_docs": [],
            "docs": [],
            "route": "general",
            "task_mode": "all",
            "st_gate": "strict",
            "mq_used": False,
            "mq_reason": None,
            "attempts": 1,
            "retry_strategy": "refine_queries",
            "guardrail_dropped_numeric": 0,
            "guardrail_dropped_anchor": 0,
            "guardrail_final_count": 1,
            "search_queries": ["q1"],
            "answer_ref_json": [{"doc_id": "doc-1", "content": "c1", "metadata": {}}],
        }


def _install_overrides(client: TestClient) -> None:
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _MinimalSearchService()
    )
    app.dependency_overrides[dependencies.get_default_llm] = lambda: SimpleNamespace()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace()
    )


def _parse_sse_final_result(raw: str) -> dict[str, Any]:
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = cast(dict[str, Any], json.loads(line[len("data: ") :]))
        if payload.get("type") == "final":
            return cast(dict[str, Any], payload["result"])
    raise AssertionError("missing final SSE payload")


def _assert_metadata_contract(
    body: dict[str, Any], *, expected_max_attempts: int
) -> None:
    metadata = cast(dict[str, Any], body["metadata"])
    required_keys = [
        "mq_mode",
        "mq_used",
        "mq_reason",
        "route",
        "st_gate",
        "attempts",
        "max_attempts",
        "retry_strategy",
        "guardrail_dropped_numeric",
        "guardrail_dropped_anchor",
        "guardrail_final_count",
        "search_queries_final",
    ]
    for key in required_keys:
        assert key in metadata

    assert isinstance(metadata["mq_mode"], str)
    assert isinstance(metadata["mq_used"], bool)
    assert metadata["mq_reason"] is None or isinstance(metadata["mq_reason"], str)
    assert metadata["route"] is None or isinstance(metadata["route"], str)
    assert metadata["st_gate"] is None or isinstance(metadata["st_gate"], str)
    assert isinstance(metadata["attempts"], int)
    assert metadata["max_attempts"] == expected_max_attempts
    assert metadata["retry_strategy"] is None or isinstance(
        metadata["retry_strategy"], str
    )
    assert isinstance(metadata["guardrail_dropped_numeric"], int)
    assert isinstance(metadata["guardrail_dropped_anchor"], int)
    assert isinstance(metadata["guardrail_final_count"], int)
    assert isinstance(metadata["search_queries_final"], list)
    assert body["search_queries"] == metadata["search_queries_final"]

    assert "index_name" in metadata
    assert isinstance(metadata["index_name"], str)

    assert "search_queries_raw" in metadata
    raw = cast(list[Any], metadata["search_queries_raw"])
    assert 0 < len(raw) <= 5
    assert all(isinstance(item, str) and item.strip() for item in raw)
    assert all(len(item) <= 120 for item in raw)


def test_agent_run_and_stream_metadata_contract(
    client: TestClient, monkeypatch
) -> None:
    _install_overrides(client)

    def _fake_new_auto_parse_agent(
        llm: Any,
        search_service: Any,
        prompt_spec: Any,
        *,
        top_k: int,
        use_canonical_retrieval: bool = False,
        event_sink: Any = None,
    ) -> _FakeAutoParseAgent:
        del llm, search_service, prompt_spec, top_k, use_canonical_retrieval
        return _FakeAutoParseAgent(event_sink=event_sink)

    monkeypatch.setattr(
        agent_router, "_new_auto_parse_agent", _fake_new_auto_parse_agent
    )

    payload = {
        "message": "metadata contract",
        "auto_parse": True,
        "max_attempts": 3,
    }

    run_resp = client.post("/api/agent/run", json=payload)
    assert run_resp.status_code == 200
    run_body = cast(dict[str, Any], run_resp.json())
    _assert_metadata_contract(run_body, expected_max_attempts=3)

    stream_resp = client.post("/api/agent/run/stream", json=payload)
    assert stream_resp.status_code == 200
    stream_body = _parse_sse_final_result(stream_resp.text)
    _assert_metadata_contract(stream_body, expected_max_attempts=3)


def test_non_issue_answer_refs_do_not_emit_issue_metadata(
    client: TestClient, monkeypatch
) -> None:
    _install_overrides(client)

    def _fake_new_auto_parse_agent(
        llm: Any,
        search_service: Any,
        prompt_spec: Any,
        *,
        top_k: int,
        use_canonical_retrieval: bool = False,
        event_sink: Any = None,
    ) -> _FakeNonIssueAnswerRefsAgent:
        del llm, search_service, prompt_spec, top_k, use_canonical_retrieval
        return _FakeNonIssueAnswerRefsAgent(event_sink=event_sink)

    monkeypatch.setattr(
        agent_router, "_new_auto_parse_agent", _fake_new_auto_parse_agent
    )

    payload = {
        "message": "non issue metadata",
        "auto_parse": True,
        "max_attempts": 3,
    }

    run_resp = client.post("/api/agent/run", json=payload)
    assert run_resp.status_code == 200
    run_body = cast(dict[str, Any], run_resp.json())
    metadata = cast(dict[str, Any], run_body["metadata"])

    assert "issue_answer_ref_count" not in metadata
    assert "issue_case_count" not in metadata
