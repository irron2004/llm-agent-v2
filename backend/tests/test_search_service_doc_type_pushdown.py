from __future__ import annotations

from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.search_service import SearchService


class _DummyCorpus:
    def __init__(self) -> None:
        # SearchService 초기화 시 EmbeddingService 생성을 피하기 위해 embedder 제공
        self.embedder = object()
        self.vector_store = None
        self.bm25_index = None


class _CapturingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, query: str, top_k: int | None = None, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return [
            RetrievalResult(
                doc_id="doc_sop",
                content="sop content",
                score=1.0,
                metadata={"doc_type": "sop"},
                raw_text="sop content",
            )
        ]


def test_search_service_passes_doc_types_to_retriever(monkeypatch) -> None:
    retriever = _CapturingRetriever()
    monkeypatch.setattr(SearchService, "_build_retriever", lambda self: retriever)

    service = SearchService(
        _DummyCorpus(),
        method="dense",
        multi_query_enabled=False,
        rerank_enabled=False,
    )

    service.search(
        "ashing rate",
        top_k=5,
        multi_query=False,
        doc_types=["sop", "setup"],
    )

    assert len(retriever.calls) == 1
    assert retriever.calls[0].get("doc_types") == ["sop", "setup"]
