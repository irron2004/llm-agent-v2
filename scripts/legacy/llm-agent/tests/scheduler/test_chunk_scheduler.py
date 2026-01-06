from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from scripts.ragflow.chunk_scheduler import (
    ChunkScheduler,
    DatasetInfo,
    JsonLineLogger,
    dry_run_summary,
    load_target_datasets,
)


class FakeDocument:
    def __init__(self, doc_id: str, name: str, states: Sequence[str]):
        self.doc_id = doc_id
        self.name = name
        self.states = [state.upper() for state in states]
        self.index = 0

    def current_state(self) -> str:
        return self.states[min(self.index, len(self.states) - 1)]

    def advance(self) -> None:
        if self.index < len(self.states) - 1:
            self.index += 1


class FakeClient:
    def __init__(self) -> None:
        self._datasets = [
            {"id": "ds_sop", "name": "sop_v3"},
            {"id": "ds_ts", "name": "ts_v3"},
        ]
        self._documents: Dict[str, Dict[str, FakeDocument]] = {
            "ds_sop": {
                "doc1": FakeDocument("doc1", "doc1", ["CANCEL", "RUNNING", "DONE"]),
            },
            "ds_ts": {
                "doc2": FakeDocument("doc2", "doc2", ["CANCEL", "RUNNING", "DONE"]),
            },
        }
        self.trigger_calls: List[Tuple[str, str]] = []

    def list_datasets(self) -> Mapping[str, Any]:
        return {"data": self._datasets}

    def get_dataset_documents(self, dataset_id: str, page: int = 1, page_size: int = 1000) -> Mapping[str, Any]:
        docs: List[Mapping[str, Any]] = []
        for doc in self._documents[dataset_id].values():
            docs.append({"id": doc.doc_id, "name": doc.name, "run": doc.current_state()})
            if doc.current_state() == "RUNNING":
                doc.advance()
        return {"data": {"docs": docs}}

    def trigger_chunk_build(self, dataset_id: str, document_ids: Iterable[str]) -> Mapping[str, Any]:
        for document_id in document_ids:
            self.trigger_calls.append((dataset_id, document_id))
            doc = self._documents[dataset_id][document_id]
            doc.advance()
        return {"code": 0}


class TimeHarness:
    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0)

    def time(self) -> float:
        return self.now


def test_load_target_datasets_preserves_order():
    client = FakeClient()
    datasets = load_target_datasets(client, ["ts_v3", "sop_v3"])
    assert [info.name for info in datasets] == ["ts_v3", "sop_v3"]


def test_chunk_scheduler_triggers_in_priority_order():
    client = FakeClient()
    time_harness = TimeHarness()
    datasets = [DatasetInfo(name="sop_v3", identifier="ds_sop"), DatasetInfo(name="ts_v3", identifier="ds_ts")]
    logger = JsonLineLogger(path=None)
    scheduler = ChunkScheduler(
        client,
        datasets,
        max_queue=1,
        poll_interval=0.0,
        states=["CANCEL", "PENDING"],
        logger=logger,
        cooldown_sec=0.0,
        post_trigger_delay=0.0,
        sleep_fn=time_harness.sleep,
        time_fn=time_harness.time,
    )

    summary = scheduler.run()

    assert client.trigger_calls == [("ds_sop", "doc1"), ("ds_ts", "doc2")]
    assert summary["triggered"]["sop_v3"] == 1
    assert summary["triggered"]["ts_v3"] == 1
    assert summary["running_total"] == 0


def test_dry_run_summary_reports_pending_counts():
    client = FakeClient()
    time_harness = TimeHarness()
    datasets = [DatasetInfo(name="sop_v3", identifier="ds_sop")]
    logger = JsonLineLogger(path=None)
    scheduler = ChunkScheduler(
        client,
        datasets,
        max_queue=5,
        poll_interval=0.0,
        states=["CANCEL"],
        logger=logger,
        cooldown_sec=0.0,
        post_trigger_delay=0.0,
        sleep_fn=time_harness.sleep,
        time_fn=time_harness.time,
    )

    summary = dry_run_summary(scheduler)

    assert summary["running_total"] == 0
    assert summary["datasets"]["sop_v3"]["pending"] == 1
    assert summary["datasets"]["sop_v3"]["total"] == 1
