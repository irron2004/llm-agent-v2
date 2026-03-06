from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import agent as agent_router


class _FakeGraph:
    def __init__(self, checkpointer: dict[str, dict[str, Any]] | None) -> None:
        self._checkpointer = checkpointer

    def get_state(self, config: dict[str, Any]) -> Any:
        thread_id = str(config["configurable"]["thread_id"])
        if not isinstance(self._checkpointer, dict):
            return None
        state = self._checkpointer.get(thread_id)
        if state is None:
            return None
        return SimpleNamespace(values=state, next=("resume",))

    def invoke(self, _command: Any, config: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(config["configurable"]["thread_id"])
        if isinstance(self._checkpointer, dict):
            self._checkpointer[thread_id] = {"resumed": True}
        return {
            "answer": f"resumed:{thread_id}",
            "judge": {},
            "docs": [],
            "search_queries": [],
        }


class _FakeLangGraphRAGAgent:
    instances: list["_FakeLangGraphRAGAgent"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.ask_user_after_retrieve = bool(
            kwargs.get("ask_user_after_retrieve", False)
        )
        self.checkpointer = kwargs.get("checkpointer")
        self._graph = _FakeGraph(
            cast(dict[str, dict[str, Any]] | None, self.checkpointer)
        )
        _FakeLangGraphRAGAgent.instances.append(self)

    def run(self, _message: str, **kwargs: Any) -> dict[str, Any]:
        thread_id = str(kwargs["thread_id"])
        if self.ask_user_after_retrieve:
            if isinstance(self.checkpointer, dict):
                self.checkpointer[thread_id] = {"pending": True}
            return {
                "answer": "",
                "judge": {},
                "docs": [],
                "search_queries": [],
                "__interrupt__": [SimpleNamespace(value={"type": "retrieval_review"})],
            }

        return {
            "answer": f"ok:{thread_id}",
            "judge": {},
            "docs": [],
            "search_queries": [],
        }


class _ResumeOnlyGraph:
    def get_state(self, _config: dict[str, Any]) -> Any:
        return SimpleNamespace(values={"pending": True}, next=("resume",))

    def invoke(self, _command: Any, config: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(config["configurable"]["thread_id"])
        return {
            "answer": f"guided-resumed:{thread_id}",
            "judge": {},
            "docs": [],
            "search_queries": [],
        }


class _ResumeOnlyAgent:
    def __init__(self) -> None:
        self._graph = _ResumeOnlyGraph()


def test_interrupt_resume_uses_fresh_agent_with_shared_checkpointer(
    client: TestClient,
    monkeypatch,
) -> None:
    _FakeLangGraphRAGAgent.instances = []
    shared_checkpointer: dict[str, dict[str, Any]] = {}

    monkeypatch.setattr(agent_router, "_checkpointer", shared_checkpointer)
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _FakeLangGraphRAGAgent)

    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: object()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: object()

    tid = "resume-regression-thread"
    first = client.post(
        "/api/agent/run",
        json={
            "message": "first",
            "thread_id": tid,
            "auto_parse": False,
            "ask_user_after_retrieve": True,
        },
    )
    assert first.status_code == 200
    first_data = cast(dict[str, Any], first.json())
    assert first_data["interrupted"] is True
    assert first_data["thread_id"] == tid

    second = client.post(
        "/api/agent/run",
        json={
            "message": "resume",
            "thread_id": tid,
            "resume_decision": True,
            "ask_user_after_retrieve": True,
            "auto_parse": False,
        },
    )
    assert second.status_code == 200
    second_data = cast(dict[str, Any], second.json())
    assert second_data["interrupted"] is False
    assert second_data["answer"] == f"resumed:{tid}"

    assert len(_FakeLangGraphRAGAgent.instances) == 2
    assert (
        _FakeLangGraphRAGAgent.instances[0] is not _FakeLangGraphRAGAgent.instances[1]
    )
    assert _FakeLangGraphRAGAgent.instances[0].checkpointer is shared_checkpointer
    assert _FakeLangGraphRAGAgent.instances[1].checkpointer is shared_checkpointer


def test_auto_parse_confirm_resume_routes_to_guided_confirm_agent(
    client: TestClient,
    monkeypatch,
) -> None:
    calls = {"guided": 0, "hil": 0}

    def _fake_guided(*_args: Any, **_kwargs: Any) -> _ResumeOnlyAgent:
        calls["guided"] += 1
        return _ResumeOnlyAgent()

    def _fake_hil(*_args: Any, **_kwargs: Any) -> _ResumeOnlyAgent:
        calls["hil"] += 1
        return _ResumeOnlyAgent()

    monkeypatch.setattr(agent_router, "_new_guided_confirm_agent", _fake_guided)
    monkeypatch.setattr(agent_router, "_new_hil_agent", _fake_hil)

    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: object()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: object()

    tid = "resume-guided-confirm-thread"
    response = client.post(
        "/api/agent/run",
        json={
            "message": "resume",
            "thread_id": tid,
            "resume_decision": {"type": "auto_parse_confirm", "confirmed": True},
            "auto_parse": True,
        },
    )

    assert response.status_code == 200
    payload = cast(dict[str, Any], response.json())
    assert payload["interrupted"] is False
    assert payload["answer"] == f"guided-resumed:{tid}"
    assert calls["guided"] == 1
    assert calls["hil"] == 0
