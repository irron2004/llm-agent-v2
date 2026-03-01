from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class _FakeLLMOutput:
    def __init__(self, text: str) -> None:
        self.text: str = text
        self.reasoning: str | None = None


class _DeterministicLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> _FakeLLMOutput:
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""

        if "router-system" in system:
            return _FakeLLMOutput("general")
        if "general-mq-system" in system:
            return _FakeLLMOutput('["pump alarm reset procedure"]')
        if "st-gate-system" in system:
            return _FakeLLMOutput("no_st")
        if "st-mq-system" in system:
            return _FakeLLMOutput('["pump alarm reset procedure"]')
        if "answer-system" in system:
            return _FakeLLMOutput("stable answer")
        if "judge-system" in system:
            if "증거(REFS): EMPTY" in user:
                return _FakeLLMOutput(
                    '{"faithful": false, "issues": ["empty"], "hint": "retry"}'
                )
            return _FakeLLMOutput('{"faithful": true, "issues": [], "hint": ""}')
        return _FakeLLMOutput("ok")


class _ScriptedSearchService:
    reranker: None = None

    def __init__(self, docs_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.docs_by_query: dict[str, list[RetrievalResult]] = docs_by_query
        self.seen_queries: list[str] = []

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        self.seen_queries.append(query)
        return list(self.docs_by_query.get(query, []))[:top_k]


def _doc(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        content=f"content-{doc_id}",
        score=score,
        metadata={"title": f"Title {doc_id}", "doc_type": "manual", "page": 1},
        raw_text=f"raw-{doc_id}",
    )


def _make_prompt_spec() -> SimpleNamespace:
    def _tmpl(system: str) -> SimpleNamespace:
        return SimpleNamespace(system=system, user="{sys.query}")

    return SimpleNamespace(
        router=_tmpl("router-system"),
        setup_mq=_tmpl("setup-mq-system"),
        ts_mq=_tmpl("ts-mq-system"),
        general_mq=_tmpl("general-mq-system"),
        st_gate=SimpleNamespace(
            system="st-gate-system",
            user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}",
        ),
        st_mq=SimpleNamespace(
            system="st-mq-system",
            user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}|{st_gate}",
        ),
        setup_ans=SimpleNamespace(
            system="answer-system", user="{sys.query}|{ref_text}"
        ),
        ts_ans=SimpleNamespace(system="answer-system", user="{sys.query}|{ref_text}"),
        general_ans=SimpleNamespace(
            system="answer-system", user="{sys.query}|{ref_text}"
        ),
        judge_setup_sys="judge-system",
        judge_ts_sys="judge-system",
        judge_general_sys="judge-system",
        auto_parse=None,
        translate=None,
    )


def _install_overrides(
    client: TestClient, *, search_service: _ScriptedSearchService
) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[dependencies.get_search_service] = lambda: search_service
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _DeterministicLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = _make_prompt_spec


def _doc_ids(body: dict[str, object]) -> list[str]:
    docs = cast(list[dict[str, object]], body["retrieved_docs"])
    return [cast(str, doc["id"]) for doc in docs]


def test_agent_run_default_fallback_is_stable_across_10_repeats(
    client: TestClient,
) -> None:
    stable_docs = [
        _doc("stable-doc-1", 1.0),
        _doc("stable-doc-2", 0.9),
        _doc("stable-doc-3", 0.8),
    ]
    search_service = _ScriptedSearchService({"pump alarm reset": stable_docs})
    _install_overrides(client, search_service=search_service)

    payload = {
        "message": "pump alarm reset",
        "auto_parse": False,
        "thread_id": "thread-stability-default",
        "max_attempts": 3,
    }

    baseline_queries: list[str] | None = None
    baseline_doc_ids: list[str] | None = None
    for _ in range(10):
        response = client.post("/api/agent/run", json=payload)
        assert response.status_code == 200
        body = cast(dict[str, object], response.json())

        metadata = cast(dict[str, object], body["metadata"])
        assert metadata["mq_mode"] == "fallback"

        current_queries = cast(list[str], body["search_queries"])
        current_doc_ids = _doc_ids(body)

        if baseline_queries is None:
            baseline_queries = current_queries
            baseline_doc_ids = current_doc_ids
        else:
            assert current_queries == baseline_queries
            assert current_doc_ids == baseline_doc_ids


def test_agent_run_empty_retrieval_triggers_fallback_mq_with_reason(
    client: TestClient,
) -> None:
    mq_docs = [_doc("mq-doc-1", 0.95), _doc("mq-doc-2", 0.9)]
    search_service = _ScriptedSearchService(
        {
            "pump alarm reset": [],
            "pump alarm reset procedure": mq_docs,
        }
    )
    _install_overrides(client, search_service=search_service)

    response = client.post(
        "/api/agent/run",
        json={
            "message": "pump alarm reset",
            "auto_parse": False,
            "thread_id": "thread-empty-fallback",
            "max_attempts": 3,
        },
    )
    assert response.status_code == 200
    body = cast(dict[str, object], response.json())
    metadata = cast(dict[str, object], body["metadata"])

    assert metadata["mq_mode"] == "fallback"
    assert metadata["mq_used"] is True
    assert metadata["mq_reason"] == "empty_retrieval"
    assert search_service.seen_queries[0] == "pump alarm reset"
    assert "pump alarm reset procedure" in search_service.seen_queries
    assert _doc_ids(body) == ["mq-doc-1", "mq-doc-2"]
