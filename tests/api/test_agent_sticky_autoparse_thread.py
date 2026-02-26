from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class _FakeLLMOutput:
    def __init__(self, text: str) -> None:
        self.text: str = text
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
    reranker: None = None

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
        ]
        return docs[:top_k]


def _fake_cache_initializer(_: object) -> SimpleNamespace:
    return SimpleNamespace(
        device_names=["SUPRA N", "OMNIS plus"],
        doc_type_names=[],
        equip_id_set=set(),
    )


def _install_overrides(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
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

    app = cast(FastAPI, client.app)
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _FakeSearchService()
    )
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _FakeLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: prompt_spec
    app.dependency_overrides[dependencies.get_reranker] = lambda: None

    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.ensure_device_cache_initialized",
        _fake_cache_initializer,
    )


def test_agent_autoparse_sticky_selections_across_turns(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_overrides(client, monkeypatch)

    first_resp = client.post(
        "/api/agent/run",
        json={
            "message": "SUPRA N 점검",
            "auto_parse": True,
            "max_attempts": 0,
            "top_k": 2,
            "use_canonical_retrieval": True,
        },
    )
    assert first_resp.status_code == 200
    first_body = cast(dict[str, object], first_resp.json())
    thread_id = cast(str, first_body["thread_id"])
    assert thread_id
    assert cast(list[str], first_body["selected_devices"]) == ["SUPRA N"]

    second_resp = client.post(
        "/api/agent/run",
        json={
            "message": "How can I reduce process drift?",
            "auto_parse": True,
            "max_attempts": 0,
            "top_k": 2,
            "thread_id": thread_id,
            "use_canonical_retrieval": True,
        },
    )
    assert second_resp.status_code == 200
    second_body = cast(dict[str, object], second_resp.json())
    assert cast(list[str], second_body["selected_devices"]) == ["SUPRA N"]

    third_resp = client.post(
        "/api/agent/run",
        json={
            "message": "OMNIS plus 점검",
            "auto_parse": True,
            "max_attempts": 0,
            "top_k": 2,
            "thread_id": thread_id,
            "use_canonical_retrieval": True,
        },
    )
    assert third_resp.status_code == 200
    third_body = cast(dict[str, object], third_resp.json())
    assert cast(list[str], third_body["selected_devices"]) == ["OMNIS plus"]
