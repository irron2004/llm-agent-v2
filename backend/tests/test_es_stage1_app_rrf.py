import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    EsSearchEngine = import_module(
        "backend.llm_infrastructure.retrieval.engines.es_search"
    ).EsSearchEngine
except ModuleNotFoundError:
    EsSearchEngine = import_module("llm_infrastructure.retrieval.engines.es_search").EsSearchEngine


class FakeEsClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"index": index, "body": body})
        return self._responses[len(self.calls) - 1]


def _make_hit(doc_id: str, chunk_id: str, page: int, score: float, content: str) -> dict[str, Any]:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "content": content,
            "page": page,
        },
    }


def test_stage1_rrf_uses_app_level_fusion_and_no_native_rrf_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dense_resp = {
        "hits": {
            "hits": [
                _make_hit("doc-1", "chunk-a", 1, 0.9, "dense-a"),
                _make_hit("doc-2", "chunk-b", 2, 0.8, "dense-b"),
                _make_hit("doc-3", "chunk-c", 3, 0.7, "dense-c"),
            ]
        }
    }
    sparse_resp = {
        "hits": {
            "hits": [
                _make_hit("doc-2", "chunk-b", 2, 5.0, "sparse-b"),
                _make_hit("doc-1", "chunk-a", 1, 4.0, "sparse-a"),
                _make_hit("doc-4", "chunk-d", 4, 3.0, "sparse-d"),
            ]
        }
    }
    fake_es = FakeEsClient([dense_resp, sparse_resp])
    engine = EsSearchEngine(es_client=cast(Any, fake_es), index_name="test-index")
    top_k = 2
    top_n = top_k * 2
    fallback_warning = "falling back to script_score"
    caplog.set_level("WARNING")

    filters = {"term": {"tenant_id.keyword": "tenant-1"}}
    hits = engine.hybrid_search(
        query_vector=[0.1, 0.2],
        query_text="vacuum leak",
        top_k=top_k,
        filters=filters,
        use_rrf=True,
        rrf_k=60,
        device_boost="etcher-01",
        device_boost_weight=3.0,
    )

    assert len(fake_es.calls) == 2
    assert len(hits) == 2
    assert [hit.doc_id for hit in hits] == ["doc-1", "doc-2"]
    assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-b"]
    assert [hit.page for hit in hits] == [1, 2]

    expected_rrf = 1.0 / 61.0 + 1.0 / 62.0
    assert math.isclose(hits[0].score, expected_rrf, rel_tol=1e-12)
    assert math.isclose(hits[1].score, expected_rrf, rel_tol=1e-12)

    first_metadata = hits[0].metadata
    second_metadata = hits[1].metadata

    for metadata in (first_metadata, second_metadata):
        assert "rrf_dense_rank" in metadata
        assert "rrf_sparse_rank" in metadata
        assert "rrf_score" in metadata
        assert "rrf_k" in metadata
        assert metadata["rrf_k"] == 60

    assert first_metadata["rrf_dense_rank"] == 1
    assert first_metadata["rrf_sparse_rank"] == 2
    assert second_metadata["rrf_dense_rank"] == 2
    assert second_metadata["rrf_sparse_rank"] == 1
    assert math.isclose(first_metadata["rrf_score"], hits[0].score, rel_tol=1e-12)
    assert math.isclose(second_metadata["rrf_score"], hits[1].score, rel_tol=1e-12)

    assert not any(fallback_warning in record.message for record in caplog.records)

    dense_body = fake_es.calls[0]["body"]
    assert "knn" in dense_body
    assert "rank" not in dense_body
    assert "sub_searches" not in dense_body
    assert dense_body["size"] == top_n
    assert dense_body["knn"]["k"] == top_n
    assert dense_body["knn"]["num_candidates"] == top_n * 2
    assert dense_body["knn"]["filter"] == filters

    sparse_body = fake_es.calls[1]["body"]
    assert "rank" not in sparse_body
    assert "sub_searches" not in sparse_body
    assert sparse_body["size"] == top_n
    assert sparse_body["query"]["bool"]["filter"] == filters
    assert "must" in sparse_body["query"]["bool"]

    fake_es_second = FakeEsClient([dense_resp, sparse_resp])
    engine_second = EsSearchEngine(
        es_client=cast(Any, fake_es_second),
        index_name="test-index",
    )
    second_hits = engine_second.hybrid_search(
        query_vector=[0.1, 0.2],
        query_text="vacuum leak",
        top_k=top_k,
        filters=filters,
        use_rrf=True,
        rrf_k=60,
        device_boost="etcher-01",
        device_boost_weight=3.0,
    )

    assert [hit.doc_id for hit in second_hits] == [hit.doc_id for hit in hits]
    assert [hit.chunk_id for hit in second_hits] == [hit.chunk_id for hit in hits]
    for idx, second_hit in enumerate(second_hits):
        assert math.isclose(second_hit.score, hits[idx].score, rel_tol=1e-12)
        assert second_hit.metadata["rrf_dense_rank"] == hits[idx].metadata["rrf_dense_rank"]
        assert second_hit.metadata["rrf_sparse_rank"] == hits[idx].metadata["rrf_sparse_rank"]
        assert math.isclose(
            second_hit.metadata["rrf_score"], hits[idx].metadata["rrf_score"], rel_tol=1e-12
        )
