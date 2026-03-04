from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from backend.api import dependencies
from backend.api.routers import agent as agent_router
from backend.config.settings import agent_settings
from backend.llm_infrastructure.llm.langgraph_agent import (
    SearchServiceRetriever,
    retrieve_node,
)
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class _RecordingSearchService:
    reranker = None

    def __init__(self) -> None:
        self.es_engine = SimpleNamespace(index_name="pe_docs_dev_current")
        self.calls: list[dict[str, Any]] = []

    def _stage1_docs(self) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                doc_id="doc-early",
                content="stage1-early",
                score=1.00,
                metadata={"title": "Early page", "doc_type": "sop", "page": 1},
                raw_text="stage1-early",
            ),
            RetrievalResult(
                doc_id="doc-deep",
                content="stage1-deep",
                score=0.99,
                metadata={"title": "Deep page", "doc_type": "sop", "page": 10},
                raw_text="stage1-deep",
            ),
        ]

    def _stage2_docs(self, doc_id: str) -> list[RetrievalResult]:
        if doc_id == "doc-early":
            return [
                RetrievalResult(
                    doc_id="doc-early",
                    content="stage2-early-p1",
                    score=0.85,
                    metadata={"title": "Early local 1", "doc_type": "sop", "page": 1},
                    raw_text="stage2-early-p1",
                ),
                RetrievalResult(
                    doc_id="doc-early",
                    content="stage2-early-p2",
                    score=0.83,
                    metadata={"title": "Early local 2", "doc_type": "sop", "page": 2},
                    raw_text="stage2-early-p2",
                ),
            ]

        return [
            RetrievalResult(
                doc_id="doc-deep",
                content="stage2-deep-p10",
                score=0.86,
                metadata={"title": "Deep local 10", "doc_type": "sop", "page": 10},
                raw_text="stage2-deep-p10",
            ),
            RetrievalResult(
                doc_id="doc-deep",
                content="stage2-deep-p11",
                score=0.84,
                metadata={"title": "Deep local 11", "doc_type": "sop", "page": 11},
                raw_text="stage2-deep-p11",
            ),
        ]

    def search(
        self, query: str, top_k: int = 10, **kwargs: object
    ) -> list[RetrievalResult]:
        doc_ids_raw = kwargs.get("doc_ids")
        doc_ids = (
            [str(v) for v in doc_ids_raw] if isinstance(doc_ids_raw, list) else None
        )
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "doc_ids": doc_ids,
            }
        )

        if doc_ids is None:
            return self._stage1_docs()[:top_k]
        if len(doc_ids) == 1:
            return self._stage2_docs(doc_ids[0])[:top_k]
        return []


class _FakeLangGraphRAGAgent:
    def __init__(
        self,
        *,
        search_service: Any,
        top_k: int,
        retrieval_top_k: int,
        **_: Any,
    ) -> None:
        self._search_service = search_service
        self._top_k = top_k
        self._retrieval_top_k = retrieval_top_k

    def run(self, query: str, **_: Any) -> dict[str, Any]:
        retriever = SearchServiceRetriever(
            self._search_service, top_k=self._retrieval_top_k
        )  # pyright: ignore[reportArgumentType]
        retrieved = retrieve_node(
            {
                "query": query,
                "search_queries": [query],
                "selected_doc_types": ["sop"],
            },
            retriever=retriever,
            reranker=None,
            retrieval_top_k=self._retrieval_top_k,
            final_top_k=self._top_k,
        )
        return {
            "answer": "ok",
            "judge": {"faithful": True, "issues": [], "hint": ""},
            "display_docs": retrieved["docs"],
            "docs": retrieved["docs"],
            "route": "general",
            "st_gate": "no_st",
            "search_queries": [query],
            "retrieval_stage2": retrieved["retrieval_stage2"],
        }


def _install_overrides(
    client: TestClient, *, search_service: _RecordingSearchService
) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[dependencies.get_search_service] = lambda: search_service
    app.dependency_overrides[dependencies.get_default_llm] = lambda: SimpleNamespace()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace()
    )


@contextmanager
def _patch_agent_settings(
    *,
    stage2_enabled: bool,
    early_page_penalty_enabled: bool,
) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                agent_settings, "second_stage_doc_retrieve_enabled", stage2_enabled
            )
        )
        stack.enter_context(patch.object(agent_settings, "second_stage_max_doc_ids", 1))
        stack.enter_context(patch.object(agent_settings, "second_stage_top_k", 2))
        stack.enter_context(
            patch.object(
                agent_settings, "early_page_penalty_enabled", early_page_penalty_enabled
            )
        )
        stack.enter_context(
            patch.object(agent_settings, "early_page_penalty_max_page", 2)
        )
        stack.enter_context(
            patch.object(agent_settings, "early_page_penalty_factor", 0.2)
        )
        yield


def _post_run(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/agent/run",
        json={
            "message": "stage2 deterministic retrieval",
            "auto_parse": False,
            "top_k": 3,
            "max_attempts": 0,
        },
    )
    assert response.status_code == 200
    return cast(dict[str, Any], response.json())


def _retrieved_doc_key_sequence(body: dict[str, Any]) -> list[tuple[str, int | None]]:
    docs = cast(list[dict[str, Any]], body["retrieved_docs"])
    return [(cast(str, d["id"]), cast(int | None, d.get("page"))) for d in docs]


def test_stage2_run_records_stage1_stage2_calls_and_metadata(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    search_service = _RecordingSearchService()
    _install_overrides(client, search_service=search_service)
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _FakeLangGraphRAGAgent)

    with _patch_agent_settings(stage2_enabled=True, early_page_penalty_enabled=False):
        body = _post_run(client)

    stage1_calls = [call for call in search_service.calls if call["doc_ids"] is None]
    stage2_calls = [
        call for call in search_service.calls if call["doc_ids"] is not None
    ]

    assert stage1_calls
    assert stage2_calls
    assert all(call["doc_ids"] is None for call in stage1_calls)
    assert all(call["doc_ids"] == ["doc-early"] for call in stage2_calls)

    metadata = cast(dict[str, Any], body["metadata"])
    retrieval_stage2 = cast(dict[str, Any], metadata["retrieval_stage2"])
    assert retrieval_stage2["enabled"] is True
    assert retrieval_stage2["doc_ids"] == ["doc-early"]


def test_stage2_metadata_shape_when_disabled(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    search_service = _RecordingSearchService()
    _install_overrides(client, search_service=search_service)
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _FakeLangGraphRAGAgent)

    with _patch_agent_settings(stage2_enabled=False, early_page_penalty_enabled=False):
        body = _post_run(client)

    metadata = cast(dict[str, Any], body["metadata"])
    retrieval_stage2 = cast(dict[str, Any], metadata["retrieval_stage2"])
    assert retrieval_stage2["enabled"] is False
    assert retrieval_stage2["doc_ids"] == []
    assert all(call["doc_ids"] is None for call in search_service.calls)


def test_early_page_penalty_can_flip_stage2_doc_selection(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _FakeLangGraphRAGAgent)

    no_penalty_search = _RecordingSearchService()
    _install_overrides(client, search_service=no_penalty_search)
    with _patch_agent_settings(stage2_enabled=True, early_page_penalty_enabled=False):
        no_penalty_body = _post_run(client)

    with_penalty_search = _RecordingSearchService()
    _install_overrides(client, search_service=with_penalty_search)
    with _patch_agent_settings(stage2_enabled=True, early_page_penalty_enabled=True):
        with_penalty_body = _post_run(client)

    no_penalty_stage2 = cast(dict[str, Any], no_penalty_body["metadata"])[
        "retrieval_stage2"
    ]
    with_penalty_stage2 = cast(dict[str, Any], with_penalty_body["metadata"])[
        "retrieval_stage2"
    ]

    assert no_penalty_stage2["doc_ids"] == ["doc-early"]
    assert with_penalty_stage2["doc_ids"] == ["doc-deep"]

    no_penalty_stage2_calls = [
        call["doc_ids"]
        for call in no_penalty_search.calls
        if call["doc_ids"] is not None
    ]
    with_penalty_stage2_calls = [
        call["doc_ids"]
        for call in with_penalty_search.calls
        if call["doc_ids"] is not None
    ]
    assert no_penalty_stage2_calls == [["doc-early"]]
    assert with_penalty_stage2_calls == [["doc-deep"]]


def test_stage2_retrieved_docs_order_is_deterministic_over_repeats(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    search_service = _RecordingSearchService()
    _install_overrides(client, search_service=search_service)
    monkeypatch.setattr(agent_router, "LangGraphRAGAgent", _FakeLangGraphRAGAgent)

    baseline: list[tuple[str, int | None]] | None = None
    with _patch_agent_settings(stage2_enabled=True, early_page_penalty_enabled=True):
        for _ in range(5):
            body = _post_run(client)
            current = _retrieved_doc_key_sequence(body)
            metadata = cast(dict[str, Any], body["metadata"])
            retrieval_stage2 = cast(dict[str, Any], metadata["retrieval_stage2"])

            assert retrieval_stage2["enabled"] is True
            assert retrieval_stage2["doc_ids"] == ["doc-deep"]

            if baseline is None:
                baseline = current
            else:
                assert current == baseline
