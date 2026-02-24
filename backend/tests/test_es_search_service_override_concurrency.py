from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeEngine:
    def __init__(self, text_fields: list[str]) -> None:
        self.text_fields = text_fields
        self.index_name = "fake-index"


class _FakeRetriever:
    def __init__(self, es_engine: _FakeEngine, barrier: threading.Barrier) -> None:
        self.es_engine = es_engine
        self.dense_weight = 0.5
        self.sparse_weight = 0.5
        self.use_rrf = True
        self.rrf_k = 60
        self._barrier = barrier

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list["_FakeResult"]:
        _ = (query, top_k, kwargs)
        self._barrier.wait(timeout=2)
        text_fields_sig = ",".join(self.es_engine.text_fields)
        doc_id = (
            f"fields={text_fields_sig}|dense={self.dense_weight}|"
            f"sparse={self.sparse_weight}|rrf={self.use_rrf}|rrf_k={self.rrf_k}"
        )
        return [_FakeResult(doc_id=doc_id, content="x", score=1.0)]


@dataclass
class _FakeResult:
    doc_id: str
    content: str
    score: float


def test_search_overrides_are_request_local_under_concurrency() -> None:
    es_service_module = importlib.import_module("backend.services.es_search_service")
    es_search_service_cls = es_service_module.EsSearchService

    barrier = threading.Barrier(2)
    engine = _FakeEngine(["base^1.0"])
    retriever = _FakeRetriever(engine, barrier)
    service = es_search_service_cls(retriever=retriever, es_engine=engine)

    def _run_a() -> str:
        result = service.search(
            "query-a",
            text_fields=["title^1.0", "summary^0.2"],
            dense_weight=0.2,
            sparse_weight=0.8,
            use_rrf=False,
            rrf_k=5,
        )
        return result[0].doc_id

    def _run_b() -> str:
        result = service.search(
            "query-b",
            text_fields=["body^1.0"],
            dense_weight=0.9,
            sparse_weight=0.1,
            use_rrf=True,
            rrf_k=99,
        )
        return result[0].doc_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_run_a)
        future_b = executor.submit(_run_b)
        doc_id_a = future_a.result(timeout=3)
        doc_id_b = future_b.result(timeout=3)

    assert doc_id_a == "fields=title^1.0,summary^0.2|dense=0.2|sparse=0.8|rrf=False|rrf_k=5"
    assert doc_id_b == "fields=body^1.0|dense=0.9|sparse=0.1|rrf=True|rrf_k=99"
