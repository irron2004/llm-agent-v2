from __future__ import annotations

from scripts.ragflow.list_datasets import (
    _document_status_counts,
    _flatten_datasets_payload,
    filter_datasets,
    summarise_dataset,
)


def test_flatten_datasets_payload_handles_nested():
    payload = {
        "code": 0,
        "data": {
            "datasets": [
                {"id": "ds1", "name": "sop_v3"},
                {"id": "ds2", "name": "ts_v3"},
            ]
        },
    }
    datasets = _flatten_datasets_payload(payload)
    assert len(datasets) == 2
    assert datasets[0]["id"] == "ds1"


def test_document_status_counts_normalises_states():
    documents = [
        {"id": "1", "run": "RUNNING"},
        {"id": "2", "run": "DONE"},
        {"id": "3", "status": "FAILED"},
        {"id": "4", "run": ""},
    ]
    counts = _document_status_counts(documents)
    assert counts["RUNNING"] == 1
    assert counts["DONE"] == 1
    assert counts["FAILED"] == 1
    assert counts["UNKNOWN"] == 1


def test_summarise_dataset_includes_counts():
    dataset = {"id": "ds1", "name": "sop_v3"}
    documents = [
        {"id": "1", "run": "RUNNING", "size": 10},
        {"id": "2", "run": "DONE", "size": 5},
    ]
    summary = summarise_dataset(dataset, documents=documents)
    assert summary["dataset_id"] == "ds1"
    assert summary["document_total"] == 2
    assert summary["document_status"]["RUNNING"] == 1
    assert summary["document_size_total"] == 15


def test_filter_datasets_matches_name_and_id():
    datasets = [
        {"id": "ds1", "name": "sop_v3"},
        {"id": "ds2", "name": "ts_v3"},
    ]
    filtered = filter_datasets(datasets, targets=["ds2"])
    assert filtered[0]["id"] == "ds2"
    filtered_by_name = filter_datasets(datasets, targets=["sop_v3"])
    assert filtered_by_name[0]["name"] == "sop_v3"
