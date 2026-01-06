from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from scripts.ragflow.chunk_scheduler import (
    DatasetInfo,
    JsonLineLogger,
    cancel_running,
)


class FakeClient:
    def __init__(self) -> None:
        self._datasets = [
            {"id": "ds_sop", "name": "sop_v3"},
            {"id": "ds_ts", "name": "ts_v3"},
        ]
        self._documents: Dict[str, List[Mapping[str, Any]]] = {
            "ds_sop": [
                {"id": "doc1", "name": "doc1", "run": "RUNNING"},
            ],
            "ds_ts": [
                {"id": "doc2", "name": "doc2", "run": "RUNNING"},
            ],
        }
        self.cancel_calls: List[tuple[str, str]] = []

    # API compatibility used by cancel_running helpers
    def get_dataset_documents(self, dataset_id: str, page: int = 1, page_size: int = 1000) -> Mapping[str, Any]:
        return {"data": {"docs": list(self._documents[dataset_id])}}

    def cancel_document(self, dataset_id: str, document_id: str) -> Mapping[str, Any]:
        self.cancel_calls.append((dataset_id, document_id))
        # Mark as cancelled to avoid double-processing in any follow-up calls
        for doc in self._documents[dataset_id]:
            if doc["id"] == document_id:
                doc["run"] = "CANCEL"
        return {"code": 0}


def test_cancel_running_cancels_in_dataset_order():
    client = FakeClient()
    datasets = [
        DatasetInfo(name="sop_v3", identifier="ds_sop"),
        DatasetInfo(name="ts_v3", identifier="ds_ts"),
    ]
    logger = JsonLineLogger(path=None)

    summary = cancel_running(
        client,
        datasets,
        states=["RUNNING"],
        max_cancel=None,
        logger=logger,
    )

    assert client.cancel_calls == [("ds_sop", "doc1"), ("ds_ts", "doc2")]
    assert summary["succeeded"]["sop_v3"] == 1
    assert summary["succeeded"]["ts_v3"] == 1


def test_cancel_running_respects_max_cancel():
    client = FakeClient()
    # Add extra running docs to exceed limit
    client._documents["ds_sop"].append({"id": "doc1b", "name": "doc1b", "run": "RUNNING"})
    client._documents["ds_ts"].append({"id": "doc2b", "name": "doc2b", "run": "RUNNING"})

    datasets = [
        DatasetInfo(name="sop_v3", identifier="ds_sop"),
        DatasetInfo(name="ts_v3", identifier="ds_ts"),
    ]
    logger = JsonLineLogger(path=None)

    summary = cancel_running(
        client,
        datasets,
        states=["RUNNING"],
        max_cancel=2,
        logger=logger,
    )

    # Only two cancels should be attempted in order
    assert client.cancel_calls[:2] == [("ds_sop", "doc1"), ("ds_sop", "doc1b")]
    assert len(client.cancel_calls) == 2
    assert (summary["attempted"]["sop_v3"] + summary["attempted"]["ts_v3"]) == 2
