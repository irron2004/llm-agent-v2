"""Agent routing and guardrail improvements unit tests."""

from __future__ import annotations

import re
from typing import Iterable, List

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import (
    load_prompt_spec,
    retrieve_node,
    route_node,
    st_mq_node,
)
from backend.llm_infrastructure.retrieval.base import RetrievalResult


def _contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


class StubLLM(BaseLLM):
    """Simple stub LLM that records key node calls."""

    def __init__(self) -> None:
        super().__init__()
        self.st_mq_called = False
        self.translate_called = 0

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs,
    ) -> LLMResponse:
        msgs = list(messages)
        system = next((m["content"] for m in msgs if m.get("role") == "system"), "")

        # Translation prompt
        if "technical translator" in system.lower():
            self.translate_called += 1
            return LLMResponse(text="한국어 번역")

        # ST MQ prompt (should be skipped when st_gate=no_st)
        if "Select and refine the final queries" in system:
            self.st_mq_called = True
            return LLMResponse(text='{"queries":["q1","q2","q3"]}')

        # Router prompt
        if "routing agent" in system.lower():
            return LLMResponse(text="general")

        return LLMResponse(text="general")


class DummyRetriever:
    """Retriever stub that records doc_type filters and returns tagged docs."""

    def __init__(self) -> None:
        self.doc_type_calls: List[List[str] | None] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        device_name: str | None = None,
        device_names: list[str] | None = None,
        doc_types: list[str] | None = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        self.doc_type_calls.append(doc_types)

        if doc_types:
            doc_id = f"biased_{len(self.doc_type_calls)}"
            doc_type = doc_types[0]
        else:
            doc_id = f"general_{len(self.doc_type_calls)}"
            doc_type = "other"

        return [
            RetrievalResult(
                doc_id=doc_id,
                content=f"content_{doc_id}",
                score=1.0,
                metadata={"doc_type": doc_type},
                raw_text=f"raw_{doc_id}",
            )
        ]


def test_route_node_general_chat_query() -> None:
    """Router should allow general chat queries to map to general."""
    spec = load_prompt_spec()
    llm = StubLLM()
    state = {"query": "너는 누구니?"}

    result = route_node(state, llm=llm, spec=spec)

    assert result["route"] == "general"


def test_st_mq_no_st_skips_llm_but_builds_bilingual_queries() -> None:
    """When st_gate=no_st, st_mq LLM call is skipped but queries are built."""
    spec = load_prompt_spec()
    llm = StubLLM()
    state = {
        "query": "Who are you?",
        "query_en": "Who are you?",
        "route": "general",
        "st_gate": "no_st",
        "general_mq_list": ["agent identity", "who are you"],
        "general_mq_ko_list": [],
    }

    result = st_mq_node(state, llm=llm, spec=spec)
    queries = result["search_queries"]

    assert llm.st_mq_called is False
    assert llm.translate_called > 0
    assert "Who are you?" in queries
    assert any(_contains_korean(q) for q in queries)


def test_retrieve_node_route_bias_merges_biased_and_general_results() -> None:
    """Route bias should not hard-filter out general docs."""
    retriever = DummyRetriever()
    state = {
        "query": "install procedure",
        "route": "setup",
        "search_queries": ["install procedure"],
        "selected_devices": [],
        "selected_doc_types": [],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=5,
        final_top_k=5,
    )

    # Route bias should trigger both doc_type-filtered and unfiltered calls.
    assert any(call for call in retriever.doc_type_calls if call)
    assert any(call is None for call in retriever.doc_type_calls)

    # General (unfiltered) docs should still be present.
    doc_types = [
        (d.metadata or {}).get("doc_type")
        for d in result["docs"]
    ]
    assert "other" in doc_types

