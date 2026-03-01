from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import agent as agent_router


class _MinimalSearchService:
    reranker = None

    def search(self, query: str, top_k: int = 10, **_: object) -> list[Any]:
        del query, top_k
        return []


class _FakeAutoParseAgent:
    def __init__(self) -> None:
        self.last_state_overrides: dict[str, Any] | None = None

    def run(self, *_: Any, **kwargs: Any) -> dict[str, Any]:
        self.last_state_overrides = cast(
            dict[str, Any] | None, kwargs.get("state_overrides")
        )
        return {
            "answer": "ok",
            "judge": {"faithful": True, "issues": [], "hint": ""},
            "display_docs": [],
            "docs": [],
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


def test_agent_run_defaults_mq_mode_to_fallback(
    client: TestClient, monkeypatch
) -> None:
    _install_overrides(client)
    fake_agent = _FakeAutoParseAgent()
    monkeypatch.setattr(agent_router.agent_settings, "mq_mode_default", "fallback")

    def _fake_new_auto_parse_agent(
        llm: Any,
        search_service: Any,
        prompt_spec: Any,
        *,
        top_k: int,
        use_canonical_retrieval: bool = False,
        event_sink: Any = None,
    ) -> _FakeAutoParseAgent:
        del llm, search_service, prompt_spec, top_k, use_canonical_retrieval, event_sink
        return fake_agent

    monkeypatch.setattr(
        agent_router, "_new_auto_parse_agent", _fake_new_auto_parse_agent
    )

    response = client.post(
        "/api/agent/run",
        json={
            "message": "default mq mode",
            "auto_parse": True,
            "max_attempts": 0,
        },
    )
    assert response.status_code == 200
    body = cast(dict[str, Any], response.json())
    assert body["metadata"]["mq_mode"] == "fallback"
    assert fake_agent.last_state_overrides is not None
    assert fake_agent.last_state_overrides["mq_mode"] == "fallback"


def test_agent_run_respects_explicit_mq_mode(client: TestClient, monkeypatch) -> None:
    _install_overrides(client)
    fake_agent = _FakeAutoParseAgent()

    def _fake_new_auto_parse_agent(
        llm: Any,
        search_service: Any,
        prompt_spec: Any,
        *,
        top_k: int,
        use_canonical_retrieval: bool = False,
        event_sink: Any = None,
    ) -> _FakeAutoParseAgent:
        del llm, search_service, prompt_spec, top_k, use_canonical_retrieval, event_sink
        return fake_agent

    monkeypatch.setattr(
        agent_router, "_new_auto_parse_agent", _fake_new_auto_parse_agent
    )

    response = client.post(
        "/api/agent/run",
        json={
            "message": "explicit mq mode",
            "auto_parse": True,
            "mq_mode": "on",
            "max_attempts": 0,
        },
    )
    assert response.status_code == 200
    body = cast(dict[str, Any], response.json())
    assert body["metadata"]["mq_mode"] == "on"
    assert fake_agent.last_state_overrides is not None
    assert fake_agent.last_state_overrides["mq_mode"] == "on"
