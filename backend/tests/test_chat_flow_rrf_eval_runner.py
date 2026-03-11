import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

runner = import_module("scripts.evaluation.run_chat_flow_retrieval_rrf_eval")


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def test_chat_flow_rrf_eval_runner_writes_expected_row(tmp_path: Path, monkeypatch: Any) -> None:
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(
        json.dumps({"qid": "qid-001", "query": "how to debug pressure drift"}) + "\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "out.jsonl"

    fake_response = {
        "interrupted": True,
        "interrupt_payload": {"type": "retrieval_review"},
        "search_queries": ["pressure drift debug"],
        "retrieved_docs": [
            {
                "id": "doc-1",
                "page": 3,
                "metadata": {
                    "doc_id": "doc-1",
                    "page": 3,
                    "chunk_id": "chunk-1",
                    "rrf_dense_rank": 1,
                    "rrf_sparse_rank": 2,
                    "rrf_score": 0.031,
                    "rrf_k": 60,
                },
            }
        ],
        "metadata": {"retrieval_debug": {"dense_count": 1, "sparse_count": 1}},
    }

    def _fake_urlopen(req: object, timeout: float) -> _FakeHttpResponse:
        _ = req
        _ = timeout
        return _FakeHttpResponse(fake_response)

    monkeypatch.setattr(runner.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_chat_flow_retrieval_rrf_eval.py",
            "--api-base-url",
            "http://localhost:8011",
            "--queries",
            str(query_file),
            "--out",
            str(out_file),
            "--limit",
            "1",
        ],
    )

    assert runner.main() == 0
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])
    assert row["qid"] == "qid-001"
    assert row["query"] == "how to debug pressure drift"
    assert row["interrupted"] is True
    assert row["interrupt_payload"]["type"] == "retrieval_review"
    assert row["search_queries"] == ["pressure drift debug"]

    docs = row["retrieved_docs"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc-1"
    assert docs[0]["page"] == 3
    assert docs[0]["metadata"]["rrf_dense_rank"] == 1
    assert docs[0]["metadata"]["rrf_sparse_rank"] == 2
    assert docs[0]["metadata"]["rrf_k"] == 60
    assert row["metadata"]["retrieval_debug"]["dense_count"] == 1
