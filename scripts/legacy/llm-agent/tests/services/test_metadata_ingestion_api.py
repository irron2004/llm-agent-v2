from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
RAGFLOW_SRC = ROOT / "ragflow"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if RAGFLOW_SRC.exists() and str(RAGFLOW_SRC) not in sys.path:
    sys.path.insert(0, str(RAGFLOW_SRC))


class _DummySwagger:  # pragma: no cover - helper for test isolation
    def __init__(self, *args, **kwargs) -> None:
        pass


class _DummyElasticsearch:  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        self._info = {"version": {"number": "8.11.3"}}

    def ping(self) -> bool:
        return True

    def info(self) -> dict[str, Any]:
        return self._info


sys.modules.setdefault("flasgger", types.SimpleNamespace(Swagger=_DummySwagger))
sys.modules.setdefault(
    "elasticsearch",
    types.SimpleNamespace(Elasticsearch=_DummyElasticsearch, NotFoundError=Exception),
)
sys.modules.setdefault(
    "elasticsearch_dsl",
    types.SimpleNamespace(UpdateByQuery=object, Q=object, Search=object, Index=object),
)
sys.modules.setdefault("elastic_transport", types.SimpleNamespace(ConnectionTimeout=Exception))

from ragflow.api.apps import chunk_app  # noqa: E402
from common.constants import RetCode


class FakeDocStore:
    def __init__(self) -> None:
        self.storage: dict[str, dict] = {}

    def seed(self, chunk: dict) -> None:
        self.storage[chunk["id"]] = copy.deepcopy(chunk)

    def get(self, chunk_id: str, *_args, **_kwargs):
        chunk = self.storage.get(chunk_id)
        return copy.deepcopy(chunk) if chunk else None

    def update(self, condition: dict, new_value: dict, *_args, **_kwargs) -> bool:
        target = condition.get("id")
        is_batch = isinstance(target, list)
        chunk_id = target[0] if isinstance(target, list) else target
        if chunk_id not in self.storage:
            return False
        if is_batch and "remove" in new_value:
            self.storage[chunk_id].pop(new_value["remove"], None)
            return True
        self.storage[chunk_id].update(new_value)
        return True


@pytest.fixture(autouse=True)
def _patch_document_service(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_doc = SimpleNamespace(id="doc-1", kb_id="kb-1", name="Demo Doc")

    def fake_get_by_id(cls, doc_id: str):
        return (True, fake_doc) if doc_id == "doc-1" else (False, None)

    def fake_get_tenant_id(cls, doc_id: str):
        return "tenant-1" if doc_id == "doc-1" else None

    monkeypatch.setattr(chunk_app.DocumentService, "get_by_id", classmethod(fake_get_by_id))
    monkeypatch.setattr(chunk_app.DocumentService, "get_tenant_id", classmethod(fake_get_tenant_id))


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeDocStore:
    store = FakeDocStore()
    store.seed(
        {
            "id": "chunk-1",
            "doc_id": "doc-1",
            "kb_id": ["kb-1"],
            "content_with_weight": "Step 3-1 local 50 Kpa",
            "q_768_vec": [0.1, 0.2],
        }
    )
    monkeypatch.setattr(chunk_app.settings, "docStoreConn", store)
    monkeypatch.setattr(chunk_app.search, "index_name", lambda tenant_id: f"idx-{tenant_id}")
    return store


@pytest.fixture
def actor() -> chunk_app.RequestActor:
    return chunk_app.RequestActor(user_id="user-1", tenant_ids={"tenant-1"})


def test_metadata_update_merges_labels_and_preserves_vectors(fake_store: FakeDocStore, actor: chunk_app.RequestActor) -> None:
    payload = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "metadata": {"doc_type": "sop", "ui_labels": ["LOCAL", "learn", "local"]},
    }

    result = chunk_app._apply_metadata_update(payload, actor=actor)

    assert "doc_type" in result["updated_keys"]
    chunk = fake_store.storage["chunk-1"]
    assert chunk["doc_type"] == "sop"
    assert chunk["ui_labels"] == ["LOCAL", "learn"]
    assert chunk["q_768_vec"] == [0.1, 0.2]


def test_replace_strategy_removes_previous_fields(fake_store: FakeDocStore, actor: chunk_app.RequestActor) -> None:
    fake_store.storage["chunk-1"].update({"doc_type": "setup", "module": "LL"})
    payload = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "metadata": {"module": "AM"},
        "merge_strategy": "replace",
    }

    result = chunk_app._apply_metadata_update(payload, actor=actor)

    assert "module" in result["updated_keys"]
    chunk = fake_store.storage["chunk-1"]
    assert chunk["module"] == "AM"
    assert "doc_type" not in chunk


def test_permission_error_raised_for_other_tenant(fake_store: FakeDocStore) -> None:
    payload = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "metadata": {"doc_type": "sop"},
    }
    bad_actor = chunk_app.RequestActor(user_id="user-2", tenant_ids={"tenant-other"})
    with pytest.raises(chunk_app.MetadataUpdateError) as err:
        chunk_app._apply_metadata_update(payload, actor=bad_actor)
    assert err.value.code == RetCode.PERMISSION_ERROR


def test_schema_validation_failure(fake_store: FakeDocStore, actor: chunk_app.RequestActor) -> None:
    payload = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "metadata": {"doc_type": "invalid"},
    }
    with pytest.raises(chunk_app.MetadataUpdateError) as err:
        chunk_app._apply_metadata_update(payload, actor=actor)
    assert err.value.code == RetCode.DATA_ERROR


def test_missing_chunk_returns_data_error(fake_store: FakeDocStore, actor: chunk_app.RequestActor) -> None:
    payload = {
        "doc_id": "doc-1",
        "chunk_id": "missing",
        "metadata": {"doc_type": "sop"},
    }
    with pytest.raises(chunk_app.MetadataUpdateError) as err:
        chunk_app._apply_metadata_update(payload, actor=actor)
    assert err.value.code == RetCode.DATA_ERROR


def test_batch_partial_failure_simulation(fake_store: FakeDocStore, actor: chunk_app.RequestActor) -> None:
    updates = [
        {"doc_id": "doc-1", "chunk_id": "chunk-1", "metadata": {"doc_type": "sop"}},
        {"doc_id": "doc-1", "chunk_id": "missing", "metadata": {"doc_type": "sop"}},
    ]

    errors: list[dict[str, Any]] = []
    success = 0
    for idx, update in enumerate(updates):
        try:
            chunk_app._apply_metadata_update(update, actor=actor)
            success += 1
        except chunk_app.MetadataUpdateError as exc:
            errors.append({"index": idx, "error_code": int(exc.code)})

    assert success == 1
    assert errors and errors[0]["index"] == 1
    assert errors[0]["error_code"] == int(RetCode.DATA_ERROR)


def test_non_embedded_chunk_is_rejected(fake_store: FakeDocStore, actor: chunk_app.RequestActor) -> None:
    # Seed a chunk without any embedding/token/body fields
    fake_store.seed({
        "id": "chunk-2",
        "doc_id": "doc-1",
        "kb_id": ["kb-1"],
    })
    payload = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-2",
        "metadata": {"doc_type": "sop"},
    }
    with pytest.raises(chunk_app.MetadataUpdateError) as err:
        chunk_app._apply_metadata_update(payload, actor=actor)
    assert err.value.code == RetCode.DATA_ERROR
