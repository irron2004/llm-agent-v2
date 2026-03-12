from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.es_chunk_v3_search_service import EsChunkV3SearchService


class _FakeEmbedder:
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        _ = texts
        return np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)


class _FakeES:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, dict[str, Any]]] = []
        self.mget_calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_calls.append((index, body))

        if index == "chunk_v3_embed_bge_m3_v1":
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk-1",
                            "_score": 9.0,
                            "_source": {
                                "chunk_id": "chunk-1",
                                "doc_id": "doc-a",
                                "doc_type": "sop",
                                "device_name": "DEV-A",
                                "equip_id": "EQ-A",
                                "lang": "ko",
                                "chapter": "chapter-a",
                                "content_hash": "hash-1",
                            },
                        },
                        {
                            "_id": "chunk-2",
                            "_score": 8.0,
                            "_source": {
                                "chunk_id": "chunk-2",
                                "doc_id": "doc-b",
                                "doc_type": "sop",
                                "device_name": "DEV-B",
                                "equip_id": "EQ-B",
                                "lang": "ko",
                                "chapter": "chapter-b",
                                "content_hash": "hash-2",
                            },
                        },
                    ]
                }
            }

        return {
            "hits": {
                "hits": [
                    {
                        "_id": "chunk-1",
                        "_score": 3.0,
                        "_source": {
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-a",
                            "page": 1,
                            "content": "raw-a",
                            "search_text": "search-a",
                            "doc_type": "sop",
                            "device_name": "DEV-A",
                            "equip_id": "EQ-A",
                            "lang": "ko",
                            "section_chapter": "10. Work Procedure",
                            "chapter_source": "title",
                            "chapter_ok": True,
                        },
                    },
                    {
                        "_id": "chunk-3",
                        "_score": 2.0,
                        "_source": {
                            "chunk_id": "chunk-3",
                            "doc_id": "doc-c",
                            "page": 3,
                            "content": "raw-c",
                            "search_text": "search-c",
                            "doc_type": "sop",
                            "device_name": "DEV-C",
                            "equip_id": "EQ-C",
                            "lang": "ko",
                            "section_chapter": "12. Check Sheet",
                            "chapter_source": "title",
                            "chapter_ok": True,
                        },
                    },
                ]
            }
        }

    def mget(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.mget_calls.append((index, body))
        by_chunk_id = {
            "chunk-1": {
                "_id": "chunk-1",
                "found": True,
                "_source": {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-a",
                    "page": 1,
                    "content": "raw-a",
                    "search_text": "search-a",
                    "doc_type": "sop",
                    "device_name": "DEV-A",
                    "equip_id": "EQ-A",
                    "lang": "ko",
                    "section_chapter": "10. Work Procedure",
                    "chapter_source": "title",
                    "chapter_ok": True,
                },
            },
            "chunk-2": {
                "_id": "chunk-2",
                "found": True,
                "_source": {
                    "chunk_id": "chunk-2",
                    "doc_id": "doc-b",
                    "page": 2,
                    "content": "raw-b",
                    "search_text": "search-b",
                    "doc_type": "sop",
                    "device_name": "DEV-B",
                    "equip_id": "EQ-B",
                    "lang": "ko",
                    "section_chapter": "11. Result",
                    "chapter_source": "title",
                    "chapter_ok": True,
                },
            },
        }
        docs = [by_chunk_id.get(cid, {"_id": cid, "found": False}) for cid in body.get("ids", [])]
        return {"docs": docs}

    def ping(self) -> bool:
        return True


def _build_service(fake_es: _FakeES) -> EsChunkV3SearchService:
    return EsChunkV3SearchService(
        es_client=fake_es,
        content_index="chunk_v3_content",
        embed_index="chunk_v3_embed_bge_m3_v1",
        embedder=_FakeEmbedder(),
        preprocessor=None,
        normalize_vectors=False,
        top_k=10,
    )


def test_v3_search_uses_embed_and_content_indices() -> None:
    fake_es = _FakeES()
    service = _build_service(fake_es)

    _ = service.search("query", top_k=3)

    dense_calls = [call for call in fake_es.search_calls if call[0] == "chunk_v3_embed_bge_m3_v1"]
    sparse_calls = [call for call in fake_es.search_calls if call[0] == "chunk_v3_content"]

    assert dense_calls
    assert sparse_calls
    assert dense_calls[0][1].get("knn") is not None
    assert sparse_calls[0][1].get("query") is not None
    assert fake_es.mget_calls
    assert fake_es.mget_calls[0][0] == "chunk_v3_content"


def test_v3_search_applies_same_filters_to_dense_and_sparse() -> None:
    fake_es = _FakeES()
    service = _build_service(fake_es)

    _ = service.search(
        "query",
        top_k=3,
        doc_types=["sop"],
        equip_ids=["eq-a"],
        lang="ko",
        device_names=["DEV-A"],
    )

    dense_body = next(
        body for index, body in fake_es.search_calls if index == "chunk_v3_embed_bge_m3_v1"
    )
    sparse_body = next(body for index, body in fake_es.search_calls if index == "chunk_v3_content")

    dense_filter = dense_body.get("knn", {}).get("filter")
    sparse_filter = sparse_body.get("query", {}).get("bool", {}).get("filter")

    assert dense_filter is not None
    assert sparse_filter is not None
    assert dense_filter == sparse_filter


def test_v3_search_joins_dense_hits_by_chunk_id() -> None:
    fake_es = _FakeES()
    service = _build_service(fake_es)

    results = service.search("query", top_k=3)

    assert results
    assert any(result.doc_id == "doc-a" for result in results)
    assert all((result.metadata or {}).get("chunk_id") for result in results)


def test_v3_search_rrf_dedupes_by_doc_id_and_chunk_id() -> None:
    fake_es = _FakeES()
    service = _build_service(fake_es)

    results = service.search("query", top_k=10, use_rrf=True, rrf_k=60)

    keys = {(result.doc_id, str((result.metadata or {}).get("chunk_id", ""))) for result in results}
    assert len(keys) == len(results)
