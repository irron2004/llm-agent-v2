from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import agent as agent_router


class _NoMutationLangGraphRAGAgent:
    instances: list["_NoMutationLangGraphRAGAgent"] = []
    records: list[dict[str, Any]] = []
    _lock = Lock()

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_locked_event_sink", False)
        object.__setattr__(self, "_event_sink", kwargs.get("event_sink"))
        object.__setattr__(self, "_locked_event_sink", True)
        object.__setattr__(self, "checkpointer", kwargs.get("checkpointer"))
        object.__setattr__(self, "_graph", self)
        with _NoMutationLangGraphRAGAgent._lock:
            _NoMutationLangGraphRAGAgent.instances.append(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "_event_sink"
            and getattr(self, "_locked_event_sink", False)
            and value is not getattr(self, "_event_sink", None)
        ):
            raise AssertionError("_event_sink must be set only in constructor")
        object.__setattr__(self, name, value)

    def run(self, message: str, **kwargs: Any) -> dict[str, Any]:
        thread_id = str(kwargs["thread_id"])
        sink = getattr(self, "_event_sink", None)
        if callable(sink):
            sink({"type": "node", "thread_id": thread_id, "message": message})
        with _NoMutationLangGraphRAGAgent._lock:
            _NoMutationLangGraphRAGAgent.records.append(
                {
                    "thread_id": thread_id,
                    "message": message,
                    "has_event_sink": callable(sink),
                }
            )
        return {
            "answer": f"ok:{thread_id}",
            "judge": {},
            "docs": [],
            "search_queries": [],
        }


def _install_overrides(client: TestClient) -> None:
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: object()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: object()


def _parse_sse_final_payload(raw: str) -> dict[str, Any]:
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = cast(dict[str, Any], json.loads(line[len("data: ") :]))
        if payload.get("type") == "final":
            return cast(dict[str, Any], payload["result"])
    raise AssertionError("missing final SSE payload")


def test_agent_run_uses_isolated_instances_without_singleton_mutation(
    client: TestClient,
    monkeypatch,
) -> None:
    _NoMutationLangGraphRAGAgent.instances = []
    _NoMutationLangGraphRAGAgent.records = []
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _NoMutationLangGraphRAGAgent)
    _install_overrides(client)

    def _call(tid: str) -> dict[str, Any]:
        resp = client.post(
            "/api/agent/run",
            json={
                "message": tid,
                "thread_id": tid,
                "auto_parse": True,
            },
        )
        assert resp.status_code == 200
        return cast(dict[str, Any], resp.json())

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(_call, "run-a")
        right = executor.submit(_call, "run-b")
        left_data = left.result()
        right_data = right.result()

    assert left_data["answer"] == "ok:run-a"
    assert right_data["answer"] == "ok:run-b"
    assert len(_NoMutationLangGraphRAGAgent.instances) == 2
    assert (
        _NoMutationLangGraphRAGAgent.instances[0]
        is not _NoMutationLangGraphRAGAgent.instances[1]
    )

    by_tid = {
        record["thread_id"]: record for record in _NoMutationLangGraphRAGAgent.records
    }
    assert by_tid["run-a"]["has_event_sink"] is False
    assert by_tid["run-b"]["has_event_sink"] is False


def test_agent_stream_sets_event_sink_in_constructor_only(
    client: TestClient,
    monkeypatch,
) -> None:
    _NoMutationLangGraphRAGAgent.instances = []
    _NoMutationLangGraphRAGAgent.records = []
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _NoMutationLangGraphRAGAgent)
    _install_overrides(client)

    def _call_stream(tid: str) -> dict[str, Any]:
        resp = client.post(
            "/api/agent/run/stream",
            json={
                "message": tid,
                "thread_id": tid,
                "auto_parse": True,
            },
        )
        assert resp.status_code == 200
        return _parse_sse_final_payload(resp.text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(_call_stream, "stream-a")
        right = executor.submit(_call_stream, "stream-b")
        left_data = left.result()
        right_data = right.result()

    assert left_data["answer"] == "ok:stream-a"
    assert right_data["answer"] == "ok:stream-b"

    stream_records = {
        record["thread_id"]: record
        for record in _NoMutationLangGraphRAGAgent.records
        if record["thread_id"].startswith("stream-")
    }
    assert stream_records["stream-a"]["has_event_sink"] is True
    assert stream_records["stream-b"]["has_event_sink"] is True
