from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    answer_node,
    issue_detail_answer_node,
)
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class SequencedLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = list(responses)

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, kwargs
        if response_model is not None:
            raise NotImplementedError
        if not self._responses:
            return LLMResponse(text="")
        return LLMResponse(text=self._responses.pop(0))


def _prompt(name: str, system: str = "", user: str = "{sys.query}\n{ref_text}") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system=system, user=user, raw={})


def _make_spec() -> PromptSpec:
    base = _prompt("base")
    return PromptSpec(
        router=_prompt("router", user="{sys.query}"),
        setup_mq=_prompt("setup_mq", user="{sys.query}"),
        ts_mq=_prompt("ts_mq", user="{sys.query}"),
        general_mq=_prompt("general_mq", user="{sys.query}"),
        st_gate=_prompt("st_gate", user="{sys.query}"),
        st_mq=_prompt("st_mq", user="{sys.query}"),
        setup_ans=base,
        ts_ans=base,
        general_ans=base,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        issue_ans=base,
        issue_detail_ans=base,
    )


def _ref(doc_id: str, doc_type: str, section: str = "detail") -> dict[str, Any]:
    return {
        "rank": 1,
        "doc_id": doc_id,
        "content": f"{doc_id} content",
        "metadata": {"doc_type": doc_type, "section": section},
    }


def _signal_docs_for_tier1() -> list[RetrievalResult]:
    docs: list[RetrievalResult] = []
    for idx in range(20):
        if idx == 0:
            score = 1.0
        elif idx == 1:
            score = 0.7
        else:
            score = 0.7 - (idx - 1) * 0.01
        if idx < 4:
            doc_type = "gcb"
            section = "detail"
        elif idx < 6:
            doc_type = "ts"
            section = "guide"
        else:
            doc_type = "myservice"
            section = "action"
        docs.append(
            RetrievalResult(
                doc_id=f"sig-{idx}",
                content=f"signal-{idx}",
                score=score,
                metadata={"doc_type": doc_type, "section": section},
                raw_text=f"signal raw {idx}",
            )
        )
    return docs


def test_issue_policy_rollout_phase1_keeps_baseline(monkeypatch: Any) -> None:
    monkeypatch.setenv("ISSUE_POLICY_ROLLOUT_PHASE", "1")

    state: dict[str, Any] = {
        "route": "general",
        "task_mode": "issue",
        "query": "alarm",
        "answer_ref_json": [
            _ref("ms-1", "myservice"),
            _ref("ms-2", "myservice"),
            _ref("ms-3", "myservice"),
            _ref("ms-4", "myservice"),
            _ref("gcb-1", "gcb"),
        ],
        "all_docs": _signal_docs_for_tier1(),
    }
    result = answer_node(state, llm=SequencedLLM(["summary"]), spec=_make_spec())

    assert result["issue_policy_rollout_phase"] == 1
    assert result["issue_policy_tier"] == "baseline"
    assert result["issue_policy_tier_shadow"] in {"tier1", "tier2", "tier3"}
    assert len(result["answer_ref_json"]) == 5
    assert result["answer_ref_json"][0]["doc_id"] == "ms-1"


def test_issue_policy_rollout_phase3_applies_live_tier_caps(monkeypatch: Any) -> None:
    monkeypatch.setenv("ISSUE_POLICY_ROLLOUT_PHASE", "3")

    state: dict[str, Any] = {
        "route": "general",
        "task_mode": "issue",
        "query": "alarm",
        "issue_case_refs": [
            _ref("ms-1", "myservice"),
            _ref("ms-2", "myservice"),
            _ref("ms-3", "myservice"),
            _ref("ms-4", "myservice"),
            _ref("ms-5", "myservice"),
            _ref("gcb-1", "gcb"),
            _ref("gcb-2", "gcb"),
            _ref("ts-1", "ts"),
            _ref("ts-2", "ts"),
            _ref("gcb-3", "gcb"),
        ],
        "all_docs": _signal_docs_for_tier1(),
    }

    result = answer_node(state, llm=SequencedLLM(["summary"]), spec=_make_spec())

    assert result["issue_policy_rollout_phase"] == 3
    assert result["issue_policy_tier"] == "tier1"
    assert result["issue_policy_tier_shadow"] == "tier1"
    assert len(result["issue_case_refs_shadow"]) <= 10
    assert len(result["answer_ref_json"]) <= 5

    myservice_answer_refs = [
        ref
        for ref in result["answer_ref_json"]
        if ref.get("metadata", {}).get("doc_type") == "myservice"
    ]
    assert len(myservice_answer_refs) <= 1


def test_issue_detail_records_ref_source_case_ref_map() -> None:
    state: dict[str, Any] = {
        "route": "general",
        "task_mode": "issue",
        "query": "alarm",
        "issue_selected_doc_id": "gcb-1",
        "answer_ref_json": [_ref("ms-1", "myservice")],
        "issue_case_ref_map": {"gcb-1": [_ref("gcb-1", "gcb")]},
    }
    result = issue_detail_answer_node(state, llm=SequencedLLM(["detail"]), spec=_make_spec())
    assert result["issue_detail_ref_source"] == "case_ref_map"
    assert result["answer_ref_json"][0]["doc_id"] == "gcb-1"
