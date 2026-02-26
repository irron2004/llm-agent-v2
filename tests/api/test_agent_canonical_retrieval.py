from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class _FakeLLMOutput:
    def __init__(self, text: str) -> None:
        self.text = text
        self.reasoning: str | None = None


class _FakeLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> _FakeLLMOutput:
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""

        if "router-system" in system:
            return _FakeLLMOutput("general")
        if (
            "setup-mq-system" in system
            or "ts-mq-system" in system
            or "general-mq-system" in system
        ):
            return _FakeLLMOutput('["mq synthetic"]')
        if "st-gate-system" in system:
            return _FakeLLMOutput("no_st")
        if "st-mq-system" in system:
            return _FakeLLMOutput('["canonical query"]')
        if "translate-system" in system:
            return _FakeLLMOutput(user.split("|", 1)[0])
        if "answer-system" in system:
            return _FakeLLMOutput("answer")
        if "judge-system" in system:
            return _FakeLLMOutput('{"faithful": true, "issues": [], "hint": ""}')
        return _FakeLLMOutput("ok")


class _FakeSearchService:
    reranker = None

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        docs = [
            RetrievalResult(
                doc_id="doc-a",
                content=f"{query}-A",
                score=0.99,
                metadata={"title": "Doc A", "doc_type": "manual", "page": 1},
                raw_text=f"{query}-A",
            ),
            RetrievalResult(
                doc_id="doc-b",
                content=f"{query}-B",
                score=0.95,
                metadata={"title": "Doc B", "doc_type": "manual", "page": 2},
                raw_text=f"{query}-B",
            ),
            RetrievalResult(
                doc_id="doc-c",
                content=f"{query}-C",
                score=0.90,
                metadata={"title": "Doc C", "doc_type": "manual", "page": 3},
                raw_text=f"{query}-C",
            ),
        ]
        return docs[:top_k]


def _install_overrides(client: TestClient) -> None:
    prompt_spec = SimpleNamespace(
        router=SimpleNamespace(system="router-system", user="{sys.query}"),
        setup_mq=SimpleNamespace(system="setup-mq-system", user="{sys.query}"),
        ts_mq=SimpleNamespace(system="ts-mq-system", user="{sys.query}"),
        general_mq=SimpleNamespace(system="general-mq-system", user="{sys.query}"),
        st_gate=SimpleNamespace(
            system="st-gate-system", user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}"
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
        auto_parse=SimpleNamespace(system="auto-parse-system", user="{query}"),
        translate=SimpleNamespace(
            system="translate-system", user="{query}|{target_language}"
        ),
        setup_ans_en=None,
        setup_ans_zh=None,
        setup_ans_ja=None,
        ts_ans_en=None,
        ts_ans_zh=None,
        ts_ans_ja=None,
        general_ans_en=None,
        general_ans_zh=None,
        general_ans_ja=None,
    )

    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _FakeSearchService()
    )
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _FakeLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: prompt_spec
    app.dependency_overrides[dependencies.get_reranker] = lambda: None


def test_agent_canonical_retrieval_returns_metadata_and_matches_retrieval_run(
    client: TestClient,
) -> None:
    _install_overrides(client)

    query = "canonical retrieval parity"
    top_k = 2

    canonical_resp = client.post(
        "/api/agent/run",
        json={
            "message": query,
            "auto_parse": False,
            "max_attempts": 0,
            "top_k": top_k,
            "use_canonical_retrieval": True,
        },
    )
    assert canonical_resp.status_code == 200
    canonical_body = cast(dict[str, Any], canonical_resp.json())

    metadata = cast(dict[str, Any], canonical_body["metadata"])
    run_id = cast(str | None, metadata.get("run_id"))
    config_hash = cast(str | None, metadata.get("effective_config_hash"))
    assert isinstance(run_id, str)
    assert len(run_id) == 32
    assert isinstance(config_hash, str)
    assert len(config_hash) == 64

    agent_doc_ids = [
        cast(str, item["id"])
        for item in cast(list[dict[str, Any]], canonical_body["retrieved_docs"])
    ]
    assert len(agent_doc_ids) == top_k

    retrieval_resp = client.post(
        "/api/retrieval/run",
        json={
            "query": query,
            "steps": ["retrieve"],
            "deterministic": True,
            "final_top_k": top_k,
        },
    )
    assert retrieval_resp.status_code == 200
    retrieval_body = cast(dict[str, Any], retrieval_resp.json())
    retrieval_doc_ids = [
        cast(str, item["doc_id"])
        for item in cast(list[dict[str, Any]], retrieval_body["docs"])
    ]

    assert agent_doc_ids == retrieval_doc_ids
