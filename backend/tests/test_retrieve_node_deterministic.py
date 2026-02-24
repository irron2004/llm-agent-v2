from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.langgraph_agent import retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class DuplicateChunkRetriever:
    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        return [
            RetrievalResult(
                doc_id="doc-a",
                content="first chunk text",
                score=1.0,
                metadata={"chunk_id": "chunk-1", "page": 10},
                raw_text="first raw",
            ),
            RetrievalResult(
                doc_id="doc-a",
                content="second chunk text",
                score=0.95,
                metadata={"chunk_id": "chunk-1", "page": 10},
                raw_text="second raw",
            ),
            RetrievalResult(
                doc_id="doc-a",
                content="third chunk text",
                score=0.9,
                metadata={"chunk_id": "chunk-2", "page": 11},
                raw_text="third raw",
            ),
        ]


class FlippingTieRetriever:
    def __init__(self) -> None:
        self._flip = False

    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        self._flip = not self._flip
        tied_docs = [
            RetrievalResult(
                doc_id="doc-b",
                content="doc b",
                score=1.0,
                metadata={"chunk_id": "2"},
                raw_text="doc b",
            ),
            RetrievalResult(
                doc_id="doc-a",
                content="doc a",
                score=1.0,
                metadata={"chunk_id": "1"},
                raw_text="doc a",
            ),
        ]
        if not self._flip:
            tied_docs.reverse()
        tied_docs.append(
            RetrievalResult(
                doc_id="doc-c",
                content="doc c",
                score=0.8,
                metadata={"chunk_id": "3"},
                raw_text="doc c",
            )
        )
        return tied_docs


class PageStringTieRetriever:
    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        return [
            RetrievalResult(
                doc_id="doc-page",
                content="page ten",
                score=1.0,
                metadata={"chunk_id": "", "page": "10"},
                raw_text="page ten",
            ),
            RetrievalResult(
                doc_id="doc-page",
                content="page two",
                score=1.0,
                metadata={"chunk_id": "", "page": "2"},
                raw_text="page two",
            ),
        ]


def test_retrieve_node_dedupes_by_stable_chunk_key() -> None:
    retriever = DuplicateChunkRetriever()
    state = {
        "query": "stable dedupe",
        "route": "general",
        "search_queries": ["stable dedupe"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    assert [doc.metadata.get("chunk_id") for doc in result["docs"]] == ["chunk-1", "chunk-2"]
    assert len(result["docs"]) == 2


def test_retrieve_node_tie_order_is_deterministic_across_repeated_calls() -> None:
    retriever = FlippingTieRetriever()
    state = {
        "query": "stable ordering",
        "route": "general",
        "search_queries": ["stable ordering"],
    }

    observed_orders = []
    for _ in range(4):
        result = retrieve_node(
            state,
            retriever=retriever,
            reranker=None,
            retrieval_top_k=3,
            final_top_k=3,
        )
        observed_orders.append([doc.doc_id for doc in result["docs"]])

    assert observed_orders == [
        ["doc-a", "doc-b", "doc-c"],
        ["doc-a", "doc-b", "doc-c"],
        ["doc-a", "doc-b", "doc-c"],
        ["doc-a", "doc-b", "doc-c"],
    ]


def test_retrieve_node_tie_break_sorts_page_strings_numerically() -> None:
    retriever = PageStringTieRetriever()
    state = {
        "query": "page tie",
        "route": "general",
        "search_queries": ["page tie"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=2,
        final_top_k=2,
    )

    assert [doc.metadata.get("page") for doc in result["docs"]] == ["2", "10"]
