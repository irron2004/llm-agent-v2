"""Tests: SOP+Setup doc_type diversity quota in retrieve_node.

When both SOP and Setup doc_types are selected, the top-k results should
include at least min_setup Setup docs and min_sop SOP docs (when available).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.settings import agent_settings
from backend.llm_infrastructure.llm.langgraph_agent import retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class SopDominantRetriever:
    """Returns 8 SOP docs and 2 setup docs, SOP scores higher."""

    def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
        docs = []
        # 8 SOP docs with high scores
        for i in range(8):
            docs.append(
                RetrievalResult(
                    doc_id=f"sop-doc-{i}",
                    content=f"sop procedure content {i}",
                    score=1.0 - i * 0.05,
                    metadata={"doc_type": "sop", "page": i + 5, "chunk_id": f"sop-{i}"},
                    raw_text=f"sop {i}",
                )
            )
        # 2 setup docs with lower scores
        for i in range(2):
            docs.append(
                RetrievalResult(
                    doc_id=f"setup-doc-{i}",
                    content=f"setup installation content {i}",
                    score=0.4 - i * 0.05,
                    metadata={"doc_type": "setup", "page": i + 1, "chunk_id": f"setup-{i}"},
                    raw_text=f"setup {i}",
                )
            )
        return docs


class SetupDominantRetriever:
    """Returns 2 SOP docs and 8 setup docs, setup scores higher."""

    def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
        docs = []
        # 8 setup docs with high scores
        for i in range(8):
            docs.append(
                RetrievalResult(
                    doc_id=f"setup-doc-{i}",
                    content=f"setup installation content {i}",
                    score=1.0 - i * 0.05,
                    metadata={"doc_type": "setup", "page": i + 5, "chunk_id": f"setup-{i}"},
                    raw_text=f"setup {i}",
                )
            )
        # 2 SOP docs with lower scores
        for i in range(2):
            docs.append(
                RetrievalResult(
                    doc_id=f"sop-doc-{i}",
                    content=f"sop procedure content {i}",
                    score=0.4 - i * 0.05,
                    metadata={"doc_type": "sop", "page": i + 1, "chunk_id": f"sop-{i}"},
                    raw_text=f"sop {i}",
                )
            )
        return docs


def _count_by_doc_type(docs: list[RetrievalResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in docs:
        dt = (d.metadata or {}).get("doc_type", "unknown")
        counts[dt] = counts.get(dt, 0) + 1
    return counts


def _make_state(query: str, doc_types: list[str]) -> dict:
    return {
        "query": query,
        "route": "setup",
        "search_queries": [query],
        "selected_doc_types": doc_types,
    }


def test_sop_dominant_setup_gets_min_quota() -> None:
    """When SOP dominates, at least min_setup setup docs should appear."""
    retriever = SopDominantRetriever()
    with patch.object(agent_settings, "doc_type_diversity_min_setup", 2):
        result = retrieve_node(
            _make_state("교체 절차", ["sop", "setup"]),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )
    counts = _count_by_doc_type(result["docs"])
    assert counts.get("setup", 0) >= 2, (
        f"Expected at least 2 setup docs, got {counts.get('setup', 0)}"
    )


def test_setup_dominant_sop_gets_min_quota() -> None:
    """When setup dominates, at least min_sop SOP docs should appear."""
    retriever = SetupDominantRetriever()
    with patch.object(agent_settings, "doc_type_diversity_min_sop", 2):
        result = retrieve_node(
            _make_state("교체 절차", ["sop", "setup"]),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )
    counts = _count_by_doc_type(result["docs"])
    assert counts.get("sop", 0) >= 2, (
        f"Expected at least 2 SOP docs, got {counts.get('sop', 0)}"
    )


def test_diversity_disabled_no_rebalance() -> None:
    """When diversity is disabled, no rebalancing should happen."""
    retriever = SopDominantRetriever()
    with (
        patch.object(agent_settings, "doc_type_diversity_enabled", False),
        patch.object(agent_settings, "doc_type_diversity_min_setup", 3),
    ):
        result = retrieve_node(
            _make_state("교체 절차", ["sop", "setup"]),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )
    counts = _count_by_doc_type(result["docs"])
    # Without diversity, setup count depends purely on score ranking
    # SOP dominates with higher scores, so setup count should be <=2 (original)
    assert counts.get("setup", 0) <= 2


def test_single_doc_type_no_quota_applied() -> None:
    """When only SOP is selected (no setup), quota should not apply."""
    retriever = SopDominantRetriever()
    result = retrieve_node(
        _make_state("교체 절차", ["sop"]),  # only SOP, no setup
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )
    counts = _count_by_doc_type(result["docs"])
    # No quota enforcement — whatever score ranking gives
    assert counts.get("sop", 0) >= 1


def test_total_count_preserved_after_rebalance() -> None:
    """Rebalancing should not change total doc count."""
    retriever = SopDominantRetriever()
    with patch.object(agent_settings, "doc_type_diversity_min_setup", 2):
        result = retrieve_node(
            _make_state("교체 절차", ["sop", "setup"]),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )
    assert len(result["docs"]) == 10
