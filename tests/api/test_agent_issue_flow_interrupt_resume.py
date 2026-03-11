from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, TypeVar, cast, overload

import pytest
from fastapi import HTTPException

from backend.api.routers import agent as agent_router
from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    Retriever,
    issue_case_selection_apply_node,
    issue_sop_confirm_apply_node,
    issue_step1_prepare_node,
    issue_step2_prepare_detail_node,
    issue_step3_sop_answer_node,
)
from backend.llm_infrastructure.retrieval.base import RetrievalResult

TModel = TypeVar("TModel")


class _StubLLM(BaseLLM):
    @overload
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[TModel],
        **kwargs: object,
    ) -> TModel: ...

    @overload
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: None = None,
        **kwargs: object,
    ) -> LLMResponse: ...

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[TModel] | None = None,
        **_: Any,
    ) -> LLMResponse | TModel:
        _ = list(messages)
        if response_model is None:
            return LLMResponse(
                text=(
                    "이슈: 압력 저하\n원인: 밸브 누설\n조치: 교체\n상태: 완료\n"
                    "REFS 발췌: SOP-ABC-001 [1]"
                )
            )
        return response_model()  # type: ignore[call-arg]


def _prompt_spec() -> SimpleNamespace:
    template = SimpleNamespace(
        system="sys", user="질문: {sys.query}\nREFS:\n{ref_text}"
    )

    return SimpleNamespace(
        issue_detail_ans=template,
        issue_detail_ans_en=None,
        issue_detail_ans_zh=None,
        issue_detail_ans_ja=None,
        issue_ans=template,
        issue_ans_en=None,
        issue_ans_zh=None,
        issue_ans_ja=None,
        general_ans=template,
        general_ans_en=None,
        general_ans_zh=None,
        general_ans_ja=None,
        setup_ans=template,
        setup_ans_en=None,
        setup_ans_zh=None,
        setup_ans_ja=None,
    )


class _EmptyRetriever:
    def retrieve(
        self, query: str, *, top_k: int = 8, **kwargs: Any
    ) -> list[RetrievalResult]:
        _ = (query, top_k, kwargs)
        return []


def _checkpoint_state(values: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(values=values)


def test_guided_resume_rejects_wrong_graph_version() -> None:
    with pytest.raises(HTTPException) as exc:
        agent_router._validate_guided_resume_checkpoint(
            thread_id="t-1",
            checkpoint_state=_checkpoint_state(
                {"graph_version": "old-version", "pending_interrupt_nonce": "nonce-1"}
            ),
            resume_decision={"nonce": "nonce-1"},
        )

    assert exc.value.status_code == 409
    assert "graph_version mismatch" in str(exc.value.detail)
    assert "expected=" in str(exc.value.detail)
    assert "actual=" in str(exc.value.detail)


def test_guided_resume_rejects_wrong_nonce() -> None:
    with pytest.raises(HTTPException) as exc:
        agent_router._validate_guided_resume_checkpoint(
            thread_id="t-2",
            checkpoint_state=_checkpoint_state(
                {
                    "graph_version": agent_router.AGENT_GRAPH_VERSION,
                    "pending_interrupt_nonce": "expected-nonce",
                }
            ),
            resume_decision={"nonce": "actual-nonce"},
        )

    assert exc.value.status_code == 409
    assert "Resume nonce mismatch" in str(exc.value.detail)


def test_guided_resume_rejects_replay_after_nonce_consumed() -> None:
    with pytest.raises(HTTPException) as exc:
        agent_router._validate_guided_resume_checkpoint(
            thread_id="t-replay",
            checkpoint_state=_checkpoint_state(
                {
                    "graph_version": agent_router.AGENT_GRAPH_VERSION,
                    "pending_interrupt_nonce": None,
                }
            ),
            resume_decision={
                "type": "issue_case_selection",
                "nonce": "used-nonce",
                "selected_doc_id": "doc-1",
            },
        )

    assert exc.value.status_code == 409
    assert "Resume nonce mismatch" in str(exc.value.detail)


def test_guided_resume_rejects_invalid_issue_case_selection_doc_id() -> None:
    with pytest.raises(HTTPException) as exc:
        agent_router._validate_guided_resume_checkpoint(
            thread_id="t-invalid-doc",
            checkpoint_state=_checkpoint_state(
                {
                    "graph_version": agent_router.AGENT_GRAPH_VERSION,
                    "pending_interrupt_nonce": "nonce-ok",
                    "issue_cases": [
                        {"doc_id": "doc-1", "title": "one", "summary": "a"},
                        {"doc_id": "doc-2", "title": "two", "summary": "b"},
                    ],
                }
            ),
            resume_decision={
                "type": "issue_case_selection",
                "nonce": "nonce-ok",
                "selected_doc_id": "doc-999",
            },
        )

    assert exc.value.status_code == 400
    assert "Invalid issue_case_selection selected_doc_id" in str(exc.value.detail)


def test_new_run_guard_rejects_pending_issue_interrupt() -> None:
    with pytest.raises(HTTPException) as exc:
        agent_router._validate_new_run_pending_interrupt_guard(
            thread_id="t-pending",
            checkpoint_state=_checkpoint_state(
                {
                    "issue_flow_step": 2,
                    "pending_interrupt_nonce": "nonce-pending",
                }
            ),
        )

    assert exc.value.status_code == 409
    assert "Pending guided interrupt exists" in str(exc.value.detail)


def test_issue_flow_prepare_apply_nodes_update_state() -> None:
    docs = [
        RetrievalResult(
            doc_id="doc-1",
            content="pressure drop case",
            score=0.9,
            metadata={"title": "Case 1"},
            raw_text="pressure drop on chamber",
        )
    ]
    prepared = issue_step1_prepare_node({"display_docs": docs})
    assert prepared["issue_flow_step"] == 1
    assert prepared["pending_interrupt_nonce"]
    assert prepared["issue_cases"][0]["doc_id"] == "doc-1"

    applied = issue_case_selection_apply_node(
        {
            "_issue_resume_decision": {
                "type": "issue_case_selection",
                "nonce": prepared["pending_interrupt_nonce"],
                "selected_doc_id": "doc-1",
            }
        }
    )
    assert applied["issue_selected_doc_id"] == "doc-1"
    assert applied["issue_flow_step"] == 2
    assert applied["pending_interrupt_nonce"] is None


def test_issue_flow_detail_and_sop_confirm_path() -> None:
    llm = _StubLLM()
    spec = _prompt_spec()
    selected_doc = RetrievalResult(
        doc_id="doc-1",
        content="case detail",
        score=0.8,
        metadata={"title": "Case 1"},
        raw_text="SOP ABC 001 관련 조치",
    )
    prepared = issue_step2_prepare_detail_node(
        {
            "query": "압력 저하",
            "issue_selected_doc_id": "doc-1",
            "docs": [selected_doc],
            "target_language": "ko",
        },
        llm=llm,
        spec=cast(PromptSpec, spec),
        doc_fetcher=None,
    )
    assert prepared["issue_flow_step"] == 2
    assert prepared["pending_interrupt_nonce"]
    assert prepared["issue_sop_candidates"]

    confirm_applied = issue_sop_confirm_apply_node(
        {"_issue_resume_decision": {"type": "issue_sop_confirm", "confirm": False}}
    )
    assert confirm_applied["issue_sop_confirmed"] is False
    assert confirm_applied["issue_flow_step"] == 3

    skip_sop = issue_step3_sop_answer_node(
        {"issue_sop_confirmed": False},
        llm=llm,
        spec=cast(PromptSpec, spec),
        retriever=cast(Retriever, _EmptyRetriever()),
    )
    assert skip_sop == {}
