from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.langgraph_agent import SearchServiceRetriever


class _ServiceWithEngine:
    def __init__(self, engine: object) -> None:
        self.es_engine = engine

    def search(self, query: str, **kwargs):
        _ = (query, kwargs)
        return []


class _ServiceWithoutEngine:
    def search(self, query: str, **kwargs):
        _ = (query, kwargs)
        return []


def test_search_service_retriever_exposes_es_engine_when_available() -> None:
    engine = object()
    retriever = SearchServiceRetriever(_ServiceWithEngine(engine), top_k=5)

    assert hasattr(retriever, "es_engine")
    assert retriever.es_engine is engine


def test_search_service_retriever_has_no_es_engine_when_unavailable() -> None:
    retriever = SearchServiceRetriever(_ServiceWithoutEngine(), top_k=5)

    assert not hasattr(retriever, "es_engine")
