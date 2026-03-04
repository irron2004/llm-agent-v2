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


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, query: str, *, top_k: int = 8, **kwargs) -> list[RetrievalResult]:
        self.calls.append(dict(kwargs))
        return [
            RetrievalResult(
                doc_id="doc-keep",
                content="kept",
                score=1.0,
                metadata={"chunk_id": "1", "doc_type": "sop", "page": 1},
                raw_text="kept",
            ),
            RetrievalResult(
                doc_id="doc-drop",
                content="drop",
                score=0.5,
                metadata={"chunk_id": "2", "doc_type": "sop", "page": 2},
                raw_text="drop",
            ),
        ]


def test_stage2_disabled_never_calls_retriever_with_doc_ids() -> None:
    retriever = RecordingRetriever()
    state = {
        "query": "sop query",
        "route": "general",
        "search_queries": ["sop query"],
        "selected_doc_types": ["sop"],
        "selected_doc_ids": ["doc-keep"],
    }

    with patch.object(agent_settings, "second_stage_doc_retrieve_enabled", False):
        result = retrieve_node(
            state,
            retriever=retriever,
            reranker=None,
            retrieval_top_k=10,
            final_top_k=10,
        )

    assert retriever.calls
    assert all("doc_ids" not in call for call in retriever.calls)
    assert result["retrieval_stage2"] == {"enabled": False, "doc_ids": []}
