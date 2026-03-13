from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from backend.api import dependencies
from backend.api.routers import agent as agent_router
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from fastapi.testclient import TestClient


class _FakeSearchService:
    reranker = None

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        del query, top_k
        return []


class _FakeRetrievalOnlyAgent:
    last_init_kwargs: dict[str, Any] = {}
    last_run_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.__class__.last_init_kwargs = kwargs

    def run(self, *_: Any, **kwargs: Any) -> dict[str, Any]:
        self.__class__.last_run_kwargs = kwargs
        docs = [
            RetrievalResult(
                doc_id="doc-all-1",
                content="retrieved snippet",
                raw_text="retrieved snippet",
                score=0.91,
                metadata={"title": "Doc All 1", "doc_type": "manual", "page": 1},
            )
        ]
        return {
            "__interrupt__": [SimpleNamespace(value={"type": "retrieval_review"})],
            "route": "general",
            "st_gate": "no_st",
            "search_queries": ["all document query"],
            "docs": docs,
            "display_docs": docs,
            "all_docs": docs,
            "judge": {},
            "detected_language": "ko",
        }


def _install_overrides(client: TestClient) -> None:
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _FakeSearchService()
    )
    app.dependency_overrides[dependencies.get_default_llm] = lambda: SimpleNamespace()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace()
    )


def test_agent_run_retrieval_only_stops_before_answer_and_uses_all_docs_mode(
    client: TestClient, monkeypatch: Any
) -> None:
    _install_overrides(client)
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _FakeRetrievalOnlyAgent)

    response = client.post(
        "/api/agent/run",
        json={
            "message": "문서 전체 검색해줘",
            "retrieval_only": True,
            "auto_parse": True,
            "max_attempts": 0,
        },
    )

    assert response.status_code == 200
    body = cast(dict[str, Any], response.json())

    assert body["interrupted"] is True
    assert cast(dict[str, Any], body["interrupt_payload"])["type"] == "retrieval_review"
    assert cast(dict[str, Any], body["metadata"])["response_mode"] == "retrieval_only"
    assert len(cast(list[dict[str, Any]], body["retrieved_docs"])) == 1

    init_kwargs = _FakeRetrievalOnlyAgent.last_init_kwargs
    assert init_kwargs["ask_user_after_retrieve"] is True
    assert init_kwargs["ask_device_selection"] is False
    assert init_kwargs["auto_parse_enabled"] is False
    assert init_kwargs["mode"] == "base"

    run_kwargs = _FakeRetrievalOnlyAgent.last_run_kwargs
    state_overrides = cast(dict[str, Any], run_kwargs["state_overrides"])
    assert "mq_mode" in state_overrides
