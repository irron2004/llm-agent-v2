from types import SimpleNamespace
from collections.abc import Mapping
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import retrieval as retrieval_router
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.retrieval_run_store import RetrievalRunSnapshotStore


class _NoOpLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> SimpleNamespace:
        del messages
        return SimpleNamespace(text="")


class _StableSearchService:
    def __init__(self) -> None:
        self._docs: list[RetrievalResult]
        self._docs = [
            RetrievalResult(
                doc_id="stable-doc-1",
                content="stable content one",
                score=1.0,
                metadata={"title": "Stable Doc 1", "doc_type": "manual", "page": 1},
                raw_text="stable content one",
            ),
            RetrievalResult(
                doc_id="stable-doc-2",
                content="stable content two",
                score=0.9,
                metadata={"title": "Stable Doc 2", "doc_type": "manual", "page": 2},
                raw_text="stable content two",
            ),
            RetrievalResult(
                doc_id="stable-doc-3",
                content="stable content three",
                score=0.8,
                metadata={"title": "Stable Doc 3", "doc_type": "manual", "page": 3},
                raw_text="stable content three",
            ),
        ]

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        del query
        return self._docs[:top_k]


class _GoldenSearchService:
    def __init__(
        self,
        corpus: dict[str, dict[str, str]],
        expected_by_query: dict[str, list[str]],
    ) -> None:
        self._corpus: dict[str, dict[str, str]] = corpus
        self._expected_by_query: dict[str, list[str]] = expected_by_query

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        doc_ids = self._expected_by_query.get(query, [])
        docs: list[RetrievalResult] = []
        for idx, doc_id in enumerate(doc_ids[:top_k]):
            item = self._corpus[doc_id]
            docs.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=item["content"],
                    score=1.0 - (idx * 0.1),
                    metadata={
                        "title": item["title"],
                        "doc_type": "manual",
                        "device_name": "etcher",
                        "equip_id": "EQ-1001",
                        "page": idx + 1,
                    },
                    raw_text=item["content"],
                )
            )
        return docs


def _install_overrides(
    client: TestClient,
    *,
    search_service: object,
    run_store: RetrievalRunSnapshotStore,
) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _NoOpLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace(translate=None)
    )
    app.dependency_overrides[dependencies.get_reranker] = lambda: None
    app.dependency_overrides[dependencies.get_search_service] = lambda: search_service
    app.dependency_overrides[retrieval_router.get_retrieval_run_store] = (
        lambda: run_store
    )


def _extract_doc_ids(payload: Mapping[str, object]) -> list[str]:
    docs_payload = payload.get("docs")
    assert isinstance(docs_payload, list)
    docs = cast(list[object], docs_payload)

    doc_ids: list[str] = []
    for item_raw in docs:
        assert isinstance(item_raw, dict)
        item = cast(dict[str, object], item_raw)
        doc_id = item.get("doc_id")
        assert isinstance(doc_id, str)
        doc_ids.append(doc_id)
    return doc_ids


@pytest.fixture
def golden_fixture() -> tuple[_GoldenSearchService, dict[str, list[str]]]:
    corpus = {
        "doc-alpha": {"title": "Alpha Guide", "content": "alpha maintenance manual"},
        "doc-beta": {"title": "Beta Guide", "content": "beta maintenance checklist"},
        "doc-gamma": {"title": "Gamma Guide", "content": "gamma alarm troubleshooting"},
        "doc-delta": {"title": "Delta Guide", "content": "delta startup procedure"},
    }
    expected_by_query = {
        "pump maintenance": ["doc-alpha", "doc-beta", "doc-delta"],
        "alarm troubleshooting": ["doc-gamma", "doc-beta"],
        "startup procedure": ["doc-delta", "doc-alpha"],
    }
    return _GoldenSearchService(corpus, expected_by_query), expected_by_query


def test_retrieval_run_deterministic_stability_n20(client: TestClient) -> None:
    _install_overrides(
        client,
        search_service=_StableSearchService(),
        run_store=RetrievalRunSnapshotStore(ttl_seconds=300),
    )

    payload = {
        "query": "stable deterministic query",
        "steps": ["retrieve"],
        "deterministic": True,
    }

    baseline_doc_ids: list[str] | None = None
    baseline_hash: str | None = None

    for _ in range(20):
        resp = client.post("/api/retrieval/run", json=payload)
        assert resp.status_code == 200
        data = cast(dict[str, object], resp.json())
        doc_ids = _extract_doc_ids(data)

        if baseline_doc_ids is None:
            baseline_doc_ids = doc_ids
            baseline_hash_payload = data.get("effective_config_hash")
            assert isinstance(baseline_hash_payload, str)
            baseline_hash = baseline_hash_payload
        else:
            assert doc_ids == baseline_doc_ids
            effective_config_hash = data.get("effective_config_hash")
            assert isinstance(effective_config_hash, str)
            assert effective_config_hash == baseline_hash


def test_retrieval_run_golden_set_exact_doc_ids(
    client: TestClient,
    golden_fixture: tuple[_GoldenSearchService, dict[str, list[str]]],
) -> None:
    search_service, expected_by_query = golden_fixture
    _install_overrides(
        client,
        search_service=search_service,
        run_store=RetrievalRunSnapshotStore(ttl_seconds=300),
    )

    for query, expected_doc_ids in expected_by_query.items():
        resp = client.post(
            "/api/retrieval/run",
            json={
                "query": query,
                "steps": ["retrieve"],
                "deterministic": True,
            },
        )
        assert resp.status_code == 200

        data = cast(dict[str, object], resp.json())
        actual_doc_ids = _extract_doc_ids(data)
        assert actual_doc_ids == expected_doc_ids
