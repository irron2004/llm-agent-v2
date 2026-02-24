from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import agent as agent_router


class _FakeLLMOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> _FakeLLMOutput:
        del messages
        return _FakeLLMOutput("ok")


class _FakeSearchService:
    def search(self, query: str, top_k: int = 10, **_: object) -> list[object]:
        del query, top_k
        return []


class _FakeAgent:
    def __init__(self, event_sink: Any = None) -> None:
        self._event_sink = event_sink

    def run(self, *_: Any, **__: Any) -> dict[str, Any]:
        if callable(self._event_sink):
            self._event_sink({"type": "node_end", "node": "retrieve"})
        return {
            "answer": "ok",
            "judge": {},
            "display_docs": [],
            "docs": [],
            "search_queries": [],
        }


def _install_retrieval_overrides(client: TestClient) -> None:
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _FakeLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace(
            router=SimpleNamespace(system="route", user="{sys.query}"), translate=None
        )
    )
    app.dependency_overrides[dependencies.get_reranker] = lambda: None
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _FakeSearchService()
    )


def _install_agent_overrides(client: TestClient, monkeypatch) -> None:
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: object()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: object()
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _FakeSearchService()
    )

    def _fake_new_auto_parse_agent(
        llm: Any,
        search_service: Any,
        prompt_spec: Any,
        *,
        top_k: int,
        event_sink: Any = None,
    ) -> _FakeAgent:
        del llm, search_service, prompt_spec, top_k
        return _FakeAgent(event_sink=event_sink)

    monkeypatch.setattr(
        agent_router, "_new_auto_parse_agent", _fake_new_auto_parse_agent
    )


def test_retrieval_trace_headers_are_echoed_and_hash_excludes_trace(
    client: TestClient,
) -> None:
    _install_retrieval_overrides(client)

    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    tracestate = "congo=t61rcWkgMzE"
    payload = {"query": "trace retrieval", "steps": ["retrieve"], "deterministic": True}

    resp_with_trace = client.post(
        "/api/retrieval/run",
        json=payload,
        headers={"traceparent": traceparent, "tracestate": tracestate},
    )
    assert resp_with_trace.status_code == 200
    body_with_trace = cast(dict[str, Any], resp_with_trace.json())
    trace = cast(dict[str, Any], body_with_trace["trace"])
    assert trace["traceparent"] == traceparent
    assert trace["tracestate"] == tracestate
    assert isinstance(trace["trace_id"], str)
    assert len(trace["trace_id"]) == 32

    resp_without_trace = client.post("/api/retrieval/run", json=payload)
    assert resp_without_trace.status_code == 200
    body_without_trace = cast(dict[str, Any], resp_without_trace.json())

    assert (
        body_without_trace["effective_config_hash"]
        == body_with_trace["effective_config_hash"]
    )


def test_retrieval_malformed_traceparent_is_treated_as_absent(
    client: TestClient,
) -> None:
    _install_retrieval_overrides(client)

    resp = client.post(
        "/api/retrieval/run",
        json={"query": "bad traceparent", "steps": ["retrieve"], "deterministic": True},
        headers={"traceparent": "malformed-traceparent", "tracestate": "ignored=value"},
    )
    assert resp.status_code == 200
    body = cast(dict[str, Any], resp.json())
    trace = cast(dict[str, Any], body["trace"])
    assert trace.get("traceparent") is None
    assert trace.get("tracestate") is None
    assert isinstance(trace["trace_id"], str)
    assert len(trace["trace_id"]) == 32


def test_agent_run_and_stream_include_trace_context(
    client: TestClient, monkeypatch
) -> None:
    _install_agent_overrides(client, monkeypatch)

    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    tracestate = "rojo=00f067aa0ba902b7"
    headers = {"traceparent": traceparent, "tracestate": tracestate}
    payload = {"message": "trace agent", "auto_parse": True, "max_attempts": 0}

    run_resp = client.post("/api/agent/run", json=payload, headers=headers)
    assert run_resp.status_code == 200
    run_body = cast(dict[str, Any], run_resp.json())
    run_trace = cast(
        dict[str, Any], cast(dict[str, Any], run_body["metadata"])["trace"]
    )
    assert run_trace["traceparent"] == traceparent
    assert run_trace["tracestate"] == tracestate
    assert isinstance(run_trace["trace_id"], str)
    assert len(run_trace["trace_id"]) == 32

    with client.stream(
        "POST", "/api/agent/run/stream", json=payload, headers=headers
    ) as stream_resp:
        assert stream_resp.status_code == 200
        open_payload: dict[str, Any] | None = None
        for line in stream_resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            event = cast(dict[str, Any], json.loads(line[6:]))
            if event.get("type") == "open":
                open_payload = event
                break
        assert open_payload is not None
        open_trace = cast(dict[str, Any], open_payload["trace"])
        assert open_trace["traceparent"] == traceparent
        assert open_trace["tracestate"] == tracestate
        assert isinstance(open_trace["trace_id"], str)
        assert len(open_trace["trace_id"]) == 32
