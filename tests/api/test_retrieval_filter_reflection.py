from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import retrieval as retrieval_router
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.retrieval_run_store import RetrievalRunSnapshotStore


class _NoOpLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> SimpleNamespace:
        del messages
        return SimpleNamespace(text="")


@dataclass
class _FilterAwareSearchService:
    def _doc(
        self, doc_id: str, *, device: str, doc_type: str, equip_id: str
    ) -> RetrievalResult:
        return RetrievalResult(
            doc_id=doc_id,
            content=f"content::{doc_id}",
            score=1.0,
            metadata={
                "title": doc_id,
                "device_name": device,
                "doc_type": doc_type,
                "equip_id": equip_id,
            },
            raw_text=f"raw::{doc_id}",
        )

    def search(
        self, query: str, top_k: int = 10, **kwargs: object
    ) -> list[RetrievalResult]:
        del query
        device_names = {
            str(v).lower() for v in cast(list[object], kwargs.get("device_names") or [])
        }
        equip_ids = {
            str(v).lower() for v in cast(list[object], kwargs.get("equip_ids") or [])
        }
        doc_types = {
            str(v).lower() for v in cast(list[object], kwargs.get("doc_types") or [])
        }

        docs = [
            self._doc(
                "doc-general", device="cleaner", doc_type="guide", equip_id="eq-10"
            ),
            self._doc(
                "doc-device-etcher", device="etcher", doc_type="guide", equip_id="eq-11"
            ),
            self._doc(
                "doc-type-manual", device="cleaner", doc_type="manual", equip_id="eq-12"
            ),
            self._doc(
                "doc-equip-99", device="cleaner", doc_type="guide", equip_id="eq-99"
            ),
        ]

        def _meta_value(doc: RetrievalResult, key: str) -> str:
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            return str(meta.get(key, "")).lower()

        if device_names:
            docs = [d for d in docs if _meta_value(d, "device_name") in device_names]
        if doc_types:
            docs = [d for d in docs if _meta_value(d, "doc_type") in doc_types]
        if equip_ids:
            docs = [d for d in docs if _meta_value(d, "equip_id") in equip_ids]

        return docs[:top_k]


def _install_overrides(client: TestClient) -> None:
    app = cast(FastAPI, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _NoOpLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = (
        lambda: SimpleNamespace(translate=None)
    )
    app.dependency_overrides[dependencies.get_reranker] = lambda: None
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _FilterAwareSearchService()
    )
    app.dependency_overrides[retrieval_router.get_retrieval_run_store] = (
        lambda: RetrievalRunSnapshotStore(ttl_seconds=300)
    )


def _doc_ids(resp_json: dict[str, Any]) -> list[str]:
    return [item["doc_id"] for item in cast(list[dict[str, Any]], resp_json["docs"])]


def test_retrieval_filters_reflect_device_doc_type_and_equip_id(
    client: TestClient,
) -> None:
    _install_overrides(client)

    base_payload = {
        "query": "filter reflection",
        "steps": ["retrieve"],
        "deterministic": True,
        "auto_parse": False,
        "skip_mq": True,
    }

    baseline = client.post("/api/retrieval/run", json=base_payload)
    assert baseline.status_code == 200
    baseline_ids = _doc_ids(cast(dict[str, Any], baseline.json()))
    assert set(baseline_ids) == {
        "doc-general",
        "doc-device-etcher",
        "doc-type-manual",
        "doc-equip-99",
    }

    device_filtered = client.post(
        "/api/retrieval/run",
        json={**base_payload, "device_names": ["etcher"]},
    )
    assert device_filtered.status_code == 200
    device_ids = _doc_ids(cast(dict[str, Any], device_filtered.json()))
    assert device_ids == ["doc-device-etcher"]
    assert device_ids != baseline_ids

    doc_type_filtered = client.post(
        "/api/retrieval/run",
        json={**base_payload, "doc_types": ["manual"], "doc_types_strict": True},
    )
    assert doc_type_filtered.status_code == 200
    doc_type_ids = _doc_ids(cast(dict[str, Any], doc_type_filtered.json()))
    assert doc_type_ids == ["doc-type-manual"]
    assert doc_type_ids != baseline_ids

    equip_filtered = client.post(
        "/api/retrieval/run",
        json={**base_payload, "equip_ids": ["eq-99"]},
    )
    assert equip_filtered.status_code == 200
    equip_ids = _doc_ids(cast(dict[str, Any], equip_filtered.json()))
    assert equip_ids == ["doc-equip-99"]
    assert equip_ids != baseline_ids
