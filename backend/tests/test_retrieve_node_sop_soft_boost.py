from __future__ import annotations
# pyright: reportMissingImports=false

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


class EqualScoreRetriever:
    """Mock retriever matching Retriever protocol (keyword-only top_k)."""

    def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                doc_id="a-ts",
                content="ts",
                score=1.0,
                metadata={"doc_type": "ts", "chunk_id": "1"},
                raw_text="ts",
            ),
            RetrievalResult(
                doc_id="z-sop",
                content="sop",
                score=1.0,
                metadata={"doc_type": "sop", "chunk_id": "2"},
                raw_text="sop",
            ),
        ]


def _make_state(doc_types: list[str]) -> AgentState:
    return {
        "query": "sop procedure",
        "route": "general",
        "search_queries": ["sop procedure"],
        "selected_doc_types": doc_types,
    }


def test_sop_soft_boost_default_reorders() -> None:
    """Default boost (1.30) multiplies SOP score and reorders above TS."""
    retriever = EqualScoreRetriever()
    result = retrieve_node(
        _make_state(["sop", "ts"]),
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    assert agent_settings.sop_soft_boost_factor == 1.30
    assert result["docs"][0].doc_id == "z-sop"
    assert result["docs"][0].score == 1.30
    assert result["docs"][1].doc_id == "a-ts"
    assert result["docs"][1].score == 1.0


def test_sop_soft_boost_uses_setting_value() -> None:
    """Prove the factor is read from settings, not hardcoded."""
    retriever = EqualScoreRetriever()
    with patch.object(agent_settings, "sop_soft_boost_factor", 2.0):
        result = retrieve_node(
            _make_state(["sop", "ts"]),
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )

    assert result["docs"][0].doc_id == "z-sop"
    assert result["docs"][0].score == 2.0


def test_no_boost_when_sop_not_selected() -> None:
    """No boost applied when SOP is not in selected_doc_types."""
    retriever = EqualScoreRetriever()
    result = retrieve_node(
        _make_state(["ts"]),
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    # Scores unchanged; alphabetical doc_id order since scores are equal
    for doc in result["docs"]:
        assert doc.score == 1.0


def test_sop_soft_boost_factor_validation() -> None:
    """Setting validates range (0, 5]."""
    import pydantic
    import pytest
    from backend.config.settings import AgentSettings

    # Valid values
    s = AgentSettings(sop_soft_boost_factor=1.0)
    assert s.sop_soft_boost_factor == 1.0

    # Invalid: zero
    with pytest.raises(pydantic.ValidationError):
        AgentSettings(sop_soft_boost_factor=0.0)

    # Invalid: negative
    with pytest.raises(pydantic.ValidationError):
        AgentSettings(sop_soft_boost_factor=-1.0)

    # Invalid: exceeds max
    with pytest.raises(pydantic.ValidationError):
        AgentSettings(sop_soft_boost_factor=6.0)
