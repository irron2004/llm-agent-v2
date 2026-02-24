from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import retrieval as retrieval_router
from backend.services.retrieval_run_store import RetrievalRunSnapshotStore


class _NoOpLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> SimpleNamespace:
        del messages
        return SimpleNamespace(text="")


class _NoOpSearchService:
    def search(self, query: str, top_k: int = 10, **_: object) -> list[object]:
        del query, top_k
        return []


def _install_overrides(
    client: TestClient, run_store: RetrievalRunSnapshotStore
) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _NoOpLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace(translate=None)
    )
    app.dependency_overrides[dependencies.get_reranker] = lambda: None
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _NoOpSearchService()
    )
    app.dependency_overrides[retrieval_router.get_retrieval_run_store] = (
        lambda: run_store
    )


def test_replay_reuses_snapshot_queries_and_doc_ids_and_sets_skip_mq(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    run_store = RetrievalRunSnapshotStore(ttl_seconds=300)
    _install_overrides(client, run_store)

    recorded_calls: list[dict[str, Any]] = []

    def _fake_pipeline(**kwargs: Any) -> dict[str, Any]:
        recorded_calls.append(kwargs)
        overrides = cast(dict[str, Any], kwargs.get("state_overrides") or {})

        queries = cast(
            list[str], overrides.get("search_queries") or [f"live::{kwargs['query']}"]
        )
        selected_doc_ids = cast(
            list[str],
            overrides.get("selected_doc_ids")
            or [f"live-doc::{kwargs['query']}", "live-doc::fallback"],
        )
        docs = [
            SimpleNamespace(
                doc_id=doc_id,
                content=f"content::{doc_id}",
                score=1.0,
                metadata={
                    "title": doc_id,
                    "doc_type": "manual",
                    "device_name": "etcher",
                },
            )
            for doc_id in selected_doc_ids
        ]

        return {
            "state": {"docs": docs, "search_queries": queries},
            "steps": {
                "retrieve": {
                    "status": "completed",
                    "artifacts": {"search_queries": queries},
                }
            },
            "executed_steps": ["retrieve"],
        }

    monkeypatch.setattr(retrieval_router, "run_retrieval_pipeline", _fake_pipeline)

    first_resp = client.post(
        "/api/retrieval/run",
        json={
            "query": "first query",
            "steps": ["retrieve"],
            "deterministic": False,
            "final_top_k": 2,
            "rerank_enabled": False,
        },
    )
    assert first_resp.status_code == 200
    first_run_id = cast(str, first_resp.json()["run_id"])

    first_snapshot_resp = client.get(f"/api/retrieval/runs/{first_run_id}")
    assert first_snapshot_resp.status_code == 200
    first_snapshot = cast(dict[str, Any], first_snapshot_resp.json())
    assert first_snapshot["search_queries"] == ["live::first query"]
    assert first_snapshot["selected_doc_ids"] == [
        "live-doc::first query",
        "live-doc::fallback",
    ]

    replay_resp = client.post(
        "/api/retrieval/run",
        json={
            "query": "replay query with different config",
            "steps": ["retrieve"],
            "deterministic": False,
            "final_top_k": 1,
            "rerank_enabled": True,
            "replay_run_id": first_run_id,
        },
    )
    assert replay_resp.status_code == 200
    replay_payload = cast(dict[str, Any], replay_resp.json())
    replay_doc_ids = [
        item["doc_id"] for item in cast(list[dict[str, Any]], replay_payload["docs"])
    ]

    assert len(recorded_calls) == 2
    replay_call = recorded_calls[1]
    replay_overrides = cast(dict[str, Any], replay_call["state_overrides"])

    assert replay_overrides["skip_mq"] is True
    assert replay_overrides["search_queries"] == ["live::first query"]
    assert replay_overrides["selected_doc_ids"] == [
        "live-doc::first query",
        "live-doc::fallback",
    ]
    assert replay_call["final_top_k"] == 1
    replay_rva = cast(dict[str, Any], replay_payload["effective_config"])[
        "request_vs_applied"
    ]
    assert cast(dict[str, Any], replay_rva)["rerank_enabled"]["requested"] is True
    assert replay_doc_ids == ["live-doc::first query", "live-doc::fallback"]
