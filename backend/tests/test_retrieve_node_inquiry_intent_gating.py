"""Tests: SOP retrieval adjustments are gated by procedural vs inquiry intent.

When SOP docs are selected but the query is an inquiry (e.g., "tool list 보여줘"),
SOP-specific penalties/boosts should NOT be applied.
When the query has procedure intent (e.g., "교체 절차"), they SHOULD apply as before.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.config.settings import agent_settings
from backend.llm_infrastructure.llm.langgraph_agent import AgentState, retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class MixedContentRetriever:
    """Returns SOP docs with scope page and procedure page at equal scores."""

    def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                doc_id="sop-scope-doc",
                content="scope of this document covers replacement procedures",
                score=1.0,
                metadata={
                    "doc_type": "sop",
                    "page": 1,
                    "section_chapter": "Scope",
                    "chunk_id": "1",
                },
                raw_text="scope",
            ),
            RetrievalResult(
                doc_id="sop-proc-doc",
                content="work procedure step 1: remove the valve",
                score=1.0,
                metadata={
                    "doc_type": "sop",
                    "page": 10,
                    "section_chapter": "Work Procedure",
                    "chunk_id": "2",
                },
                raw_text="procedure",
            ),
            RetrievalResult(
                doc_id="sop-tool-doc",
                content="tool list: wrench 10mm, torque driver",
                score=1.0,
                metadata={
                    "doc_type": "sop",
                    "page": 3,
                    "section_chapter": "Tool List",
                    "chunk_id": "3",
                },
                raw_text="tool list",
            ),
        ]


def _make_state(query: str, route: str = "general") -> AgentState:
    return {
        "query": query,
        "route": route,
        "search_queries": [query],
        "selected_doc_types": ["sop"],
    }


# --- Inquiry queries: SOP penalties should NOT apply ---


def test_inquiry_query_no_scope_penalty() -> None:
    """Inquiry query ('tool list 보여줘') should not penalize scope/early pages."""
    retriever = MixedContentRetriever()
    result = retrieve_node(
        _make_state("tool list 보여줘", route="general"),
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )
    docs = result["docs"]
    # All scores should remain 1.0 (no penalty applied)
    for doc in docs:
        assert doc.score == 1.0, (
            f"Inquiry query should not penalize: doc page={doc.metadata.get('page')}, "
            f"chapter={doc.metadata.get('section_chapter')}, score={doc.score}"
        )


def test_inquiry_query_no_early_page_penalty() -> None:
    """Inquiry query should not apply early page penalty."""
    retriever = MixedContentRetriever()
    with patch.object(agent_settings, "early_page_penalty_enabled", True):
        result = retrieve_node(
            _make_state("scope 알려줘", route="general"),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )
    docs = result["docs"]
    scope_doc = next(d for d in docs if d.metadata.get("section_chapter") == "Scope")
    assert scope_doc.score == 1.0, (
        f"Inquiry query should not apply early page penalty: score={scope_doc.score}"
    )


def test_inquiry_query_no_sop_soft_boost() -> None:
    """Inquiry query should not apply SOP soft boost."""

    class SopTsMixRetriever:
        def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    doc_id="ts-doc",
                    content="troubleshooting guide",
                    score=1.0,
                    metadata={"doc_type": "ts", "chunk_id": "1"},
                    raw_text="ts",
                ),
                RetrievalResult(
                    doc_id="sop-doc",
                    content="worksheet overview",
                    score=1.0,
                    metadata={"doc_type": "sop", "chunk_id": "2"},
                    raw_text="sop",
                ),
            ]

    retriever = SopTsMixRetriever()
    result = retrieve_node(
        _make_state("worksheet 조회", route="general"),
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )
    # Both docs should keep score=1.0 (no SOP soft boost)
    for doc in result["docs"]:
        assert doc.score == 1.0, (
            f"Inquiry query should not boost SOP: doc_id={doc.doc_id}, score={doc.score}"
        )


# --- Procedure queries: SOP penalties SHOULD apply as before ---


def test_procedure_query_scope_penalty_applies() -> None:
    """Procedure query ('교체 절차') should penalize scope pages."""
    retriever = MixedContentRetriever()
    result = retrieve_node(
        _make_state("slot valve 교체 절차", route="setup"),
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )
    docs = result["docs"]
    scope_doc = next(d for d in docs if d.metadata.get("section_chapter") == "Scope")
    proc_doc = next(d for d in docs if d.metadata.get("section_chapter") == "Work Procedure")
    # Scope should be penalized, procedure should be boosted
    assert scope_doc.score < 1.0, "Scope page should be penalized for procedure query"
    assert proc_doc.score > 1.0, "Work Procedure should be boosted for procedure query"


def test_procedure_query_early_page_penalty_applies() -> None:
    """Procedure query should apply early page penalty."""
    retriever = MixedContentRetriever()
    with patch.object(agent_settings, "early_page_penalty_enabled", True):
        result = retrieve_node(
            _make_state("sensor board 교체 방법", route="setup"),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )
    docs = result["docs"]
    scope_doc = next(d for d in docs if d.metadata.get("section_chapter") == "Scope")
    # Page 1 should be penalized
    assert scope_doc.score < 1.0, "Early page should be penalized for procedure query"


# --- Procedure wins rule ---


def test_procedure_wins_over_inquiry() -> None:
    """When both procedure and inquiry keywords present, procedure wins."""
    retriever = MixedContentRetriever()
    result = retrieve_node(
        _make_state("교체 절차 조회해줘", route="setup"),
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )
    docs = result["docs"]
    proc_doc = next(d for d in docs if d.metadata.get("section_chapter") == "Work Procedure")
    # Procedure boost should apply (procedure wins)
    assert proc_doc.score > 1.0, "Procedure wins: boost should apply despite inquiry keyword"
