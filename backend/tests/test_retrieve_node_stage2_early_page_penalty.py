from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import agent_settings
from backend.llm_infrastructure.llm.langgraph_agent import retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class Stage2OnlyRetriever:
    def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
        if "doc_ids" not in kwargs:
            return []
        doc_id = kwargs["doc_ids"][0]
        return [
            RetrievalResult(
                doc_id=doc_id,
                content="early page",
                score=1.0,
                metadata={"doc_type": "sop", "page": 1, "chunk_id": "1"},
                raw_text="early page",
            ),
            RetrievalResult(
                doc_id=doc_id,
                content="deep page",
                score=0.8,
                metadata={"doc_type": "sop", "page": 5, "chunk_id": "2"},
                raw_text="deep page",
            ),
        ]


def test_stage2_per_call_early_page_penalty_reorders_before_rrf_merge() -> None:
    retriever = Stage2OnlyRetriever()
    # 절차 의도(procedure intent)가 있어야 early_page_penalty가 적용됨
    state = {
        "query": "sop 교체 절차",
        "route": "setup",
        "search_queries": ["sop 교체 절차"],
        "selected_doc_types": ["sop"],
        "selected_doc_ids": ["doc-stage2"],
    }

    with (
        patch.object(agent_settings, "second_stage_doc_retrieve_enabled", True),
        patch.object(agent_settings, "second_stage_max_doc_ids", 1),
        patch.object(agent_settings, "second_stage_top_k", 10),
        patch.object(agent_settings, "early_page_penalty_enabled", False),
        patch.object(agent_settings, "sop_soft_boost_factor", 1.0),
    ):
        without_penalty = retrieve_node(
            state,
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )

    with (
        patch.object(agent_settings, "second_stage_doc_retrieve_enabled", True),
        patch.object(agent_settings, "second_stage_max_doc_ids", 1),
        patch.object(agent_settings, "second_stage_top_k", 10),
        patch.object(agent_settings, "early_page_penalty_enabled", True),
        patch.object(agent_settings, "early_page_penalty_max_page", 2),
        patch.object(agent_settings, "early_page_penalty_factor", 0.3),
        patch.object(agent_settings, "sop_soft_boost_factor", 1.0),
    ):
        with_penalty = retrieve_node(
            state,
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )

    assert without_penalty["docs"][0].metadata.get("page") == 1
    assert with_penalty["docs"][0].metadata.get("page") == 5
