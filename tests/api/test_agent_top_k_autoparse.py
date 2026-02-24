from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import agent as agent_router
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class _MinimalSearchService:
    reranker = None

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        return []


class _FakeAutoParseAgent:
    def __init__(self, top_k: int) -> None:
        self._top_k = top_k

    def run(self, *_: Any, **__: Any) -> dict[str, Any]:
        docs = [
            RetrievalResult(
                doc_id=f"doc-{idx}",
                content=f"content-{idx}",
                score=1.0 - (idx * 0.01),
                metadata={"title": f"Title {idx}", "page": idx + 1},
                raw_text=f"raw-{idx}",
            )
            for idx in range(self._top_k)
        ]
        return {
            "answer": f"answer-top-k-{self._top_k}",
            "judge": {"faithful": True, "issues": [], "hint": ""},
            "display_docs": docs,
            "docs": docs,
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


def test_agent_run_autoparse_honors_request_top_k(
    client: TestClient, monkeypatch
) -> None:
    _install_overrides(client)

    def _fake_new_auto_parse_agent(
        llm: Any,
        search_service: Any,
        prompt_spec: Any,
        *,
        top_k: int,
        event_sink: Any = None,
    ) -> _FakeAutoParseAgent:
        del llm, search_service, prompt_spec, event_sink
        return _FakeAutoParseAgent(top_k=top_k)

    monkeypatch.setattr(
        agent_router, "_new_auto_parse_agent", _fake_new_auto_parse_agent
    )

    resp_top_5 = client.post(
        "/api/agent/run",
        json={
            "message": "auto parse top_k 5",
            "auto_parse": True,
            "top_k": 5,
            "max_attempts": 0,
        },
    )
    assert resp_top_5.status_code == 200
    body_top_5 = cast(dict[str, Any], resp_top_5.json())
    docs_top_5 = cast(list[dict[str, Any]], body_top_5["retrieved_docs"])
    assert 0 < len(docs_top_5) <= 5

    resp_top_20 = client.post(
        "/api/agent/run",
        json={
            "message": "auto parse top_k 20",
            "auto_parse": True,
            "top_k": 20,
            "max_attempts": 0,
        },
    )
    assert resp_top_20.status_code == 200
    body_top_20 = cast(dict[str, Any], resp_top_20.json())
    docs_top_20 = cast(list[dict[str, Any]], body_top_20["retrieved_docs"])
    assert 0 < len(docs_top_20) <= 20
    assert len(docs_top_20) > len(docs_top_5)
