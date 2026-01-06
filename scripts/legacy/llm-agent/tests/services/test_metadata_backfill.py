from __future__ import annotations

from typing import Any, Dict, List

from scripts.ingestion.augment_metadata import MetadataBackfillJob
from ragflow.rag.flow.normalizer import MetadataNormalizer


def test_metadata_normalizer_finds_units_and_labels() -> None:
    text = "Step 3-1 local pressure is 50 Kpa"
    normalized = MetadataNormalizer.normalize_text(text)

    assert "[Step 3-1]" in normalized["normalized_text"]
    assert normalized["ui_labels"] == ["LOCAL"]
    assert normalized["units"][0]["unit"] == "kPa"


class StubClient:
    def __init__(self) -> None:
        self.sent_batches: List[List[Dict[str, Any]]] = []

    def list_documents(self, dataset_id: str, page: int, page_size: int) -> list[dict[str, Any]]:
        if page > 1:
            return []
        return [{"id": "doc-1"}]

    def list_chunks(self, dataset_id: str, document_id: str, page: int, page_size: int) -> list[dict[str, Any]]:
        if page > 1:
            return []
        return [{"id": "chunk-1", "content": "Step 2-3 learn 40 mtorr"}]

    def send_batch(self, updates: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
        self.sent_batches.append(updates)
        return {"success_count": len(updates), "error_count": 0, "errors": []}


def test_backfill_job_builds_valid_updates() -> None:
    client = StubClient()
    job = MetadataBackfillJob(client, batch_size=1)
    job.run(dataset_id="kb-1", dry_run=False, max_documents=1, max_chunks=1)

    assert client.sent_batches, "Expected at least one batch to be sent"
    update = client.sent_batches[0][0]
    metadata = update["metadata"]
    assert metadata["ui_labels"] == ["LEARN"]
    assert metadata["units"][0]["unit"] == "mTorr"
