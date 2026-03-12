"""Phase 4: ES 적재 + 동기화 검증 스크립트.

chunk_v3_content 인덱스에 원문+메타를, chunk_v3_embed_{model}_v1 인덱스에 벡터를 적재한다.

Usage:
    # Content 인덱스 적재
    python scripts/chunk_v3/run_ingest.py content \
        --chunks data/chunks_v3/all_chunks.jsonl

    # Embed 인덱스 적재
    python scripts/chunk_v3/run_ingest.py embed \
        --model bge_m3 \
        --embeddings data/chunks_v3/embeddings_bge_m3.npy \
        --chunk-ids data/chunks_v3/chunk_ids_bge_m3.jsonl

    # 동기화 검증
    python scripts/chunk_v3/run_ingest.py verify \
        --model bge_m3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.common import load_chunks_jsonl
from scripts.chunk_v3.run_embedding import MODEL_CONFIGS

CONTENT_INDEX = "chunk_v3_content"


def build_embed_index_name(model_key: str) -> str:
    return f"chunk_v3_embed_{model_key}_v1"


def _get_es_client():
    """Elasticsearch 클라이언트 생성."""
    from backend.config.settings import search_settings
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = {"hosts": [search_settings.es_host]}
    if search_settings.es_user and search_settings.es_password:
        kwargs["basic_auth"] = (search_settings.es_user, search_settings.es_password)
    kwargs["verify_certs"] = bool(getattr(search_settings, "es_verify_certs", True))

    kwargs.setdefault("request_timeout", 120)
    return Elasticsearch(**kwargs)


def _get_manager():
    """EsIndexManager 인스턴스 생성."""
    from backend.config.settings import search_settings
    from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager

    return EsIndexManager(
        es_host=search_settings.es_host,
        env=search_settings.es_env,
        es_user=search_settings.es_user
        if hasattr(search_settings, "es_user")
        else None,
        es_password=search_settings.es_password
        if hasattr(search_settings, "es_password")
        else None,
    )


def _build_chunk_v3_settings() -> dict[str, Any]:
    from backend.llm_infrastructure.elasticsearch.mappings import get_index_settings

    settings = get_index_settings()
    settings["index.mapping.total_fields.limit"] = 2000
    return settings


def _prepare_index(
    es,
    index_name: str,
    body: dict[str, Any],
    recreate: bool,
) -> None:
    if recreate and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Deleted index: {index_name}")

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=body)
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")


def _set_refresh_interval(es, index_name: str, interval: str) -> None:
    es.indices.put_settings(
        index=index_name,
        body={"index": {"refresh_interval": interval}},
    )


def _run_streaming_bulk(
    es,
    actions: Iterable[dict[str, Any]],
    chunk_size: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    from elasticsearch.helpers import streaming_bulk

    success = 0
    failed = 0
    sampled_errors: list[dict[str, Any]] = []
    for ok, item in streaming_bulk(
        es,
        actions,
        chunk_size=chunk_size,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=8.0,
        retry_on_status=(429, 500, 502, 503, 504),
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            success += 1
        else:
            failed += 1
            if len(sampled_errors) < 5:
                sampled_errors.append(item)
    return success, failed, sampled_errors


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest_content(
    chunks_path: str,
    content_index: str = CONTENT_INDEX,
    batch_size: int = 500,
    recreate: bool = False,
) -> None:
    """chunk_v3_content에 원문+메타 적재 (embedding 없음).

    Args:
        chunks_path: JSONL 파일 경로
        batch_size: bulk 배치 크기
    """
    from backend.llm_infrastructure.elasticsearch.mappings import (
        get_chunk_v3_content_mapping,
    )

    es = _get_es_client()
    chunks = load_chunks_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    body = {
        "settings": _build_chunk_v3_settings(),
        "mappings": get_chunk_v3_content_mapping(),
    }
    _prepare_index(es, content_index, body=body, recreate=recreate)

    allowed_fields = {
        "chunk_id",
        "doc_id",
        "page",
        "lang",
        "content",
        "search_text",
        "chunk_summary",
        "chunk_keywords",
        "doc_type",
        "device_name",
        "equip_id",
        "tenant_id",
        "project_id",
        "chapter",
        "section_chapter",
        "section_number",
        "chapter_source",
        "chapter_ok",
        "content_hash",
        "chunk_version",
        "pipeline_version",
        "extra_meta",
        "created_at",
    }
    created_at = datetime.now(timezone.utc).isoformat()

    def _gen_actions():
        for chunk in chunks:
            doc = asdict(chunk)
            unknown = set(doc.keys()) - allowed_fields
            if unknown:
                raise ValueError(
                    f"Unexpected top-level chunk fields: {sorted(unknown)} in {chunk.chunk_id}"
                )
            extra_meta = (
                doc.get("extra_meta") if isinstance(doc.get("extra_meta"), dict) else {}
            )
            raw_keywords = extra_meta.get("chunk_keywords", [])
            chunk_keywords = (
                [
                    str(keyword).strip()
                    for keyword in raw_keywords
                    if str(keyword).strip()
                ]
                if isinstance(raw_keywords, list)
                else []
            )

            doc["chunk_summary"] = str(extra_meta.get("chunk_summary", "") or "")
            doc["chunk_keywords"] = chunk_keywords
            doc["tenant_id"] = str(extra_meta.get("tenant_id", "") or "")
            doc["project_id"] = str(extra_meta.get("project_id", "") or "")
            doc["created_at"] = created_at
            yield {
                "_index": content_index,
                "_id": chunk.chunk_id,
                "_source": doc,
            }

    _set_refresh_interval(es, content_index, "-1")
    try:
        success, failed, sampled_errors = _run_streaming_bulk(
            es, _gen_actions(), chunk_size=batch_size
        )
    finally:
        _set_refresh_interval(es, content_index, "1s")

    print(f"Indexed {success} documents to {content_index}")
    if failed:
        print(f"  Failed: {failed}")
        for err in sampled_errors:
            print(f"    {err}")


def ingest_embeddings(
    model_key: str,
    embeddings_path: str,
    chunk_ids_path: str,
    chunks_path: str,
    embed_index: str | None = None,
    batch_size: int = 500,
    recreate: bool = False,
) -> None:
    """chunk_v3_embed_{model}_v1에 벡터 적재.

    Args:
        model_key: 모델 키 (MODEL_CONFIGS)
        embeddings_path: .npy 파일 경로
        chunk_ids_path: chunk_ids JSONL 파일 경로
        batch_size: bulk 배치 크기
    """
    from backend.llm_infrastructure.elasticsearch.mappings import (
        get_chunk_v3_embed_mapping,
    )

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    cfg = MODEL_CONFIGS[model_key]
    index_name = embed_index or build_embed_index_name(model_key)

    es = _get_es_client()

    vectors = np.load(embeddings_path)
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got {vectors.shape}")

    expected_dims = int(cfg["dims"])
    if expected_dims > 4096:
        raise ValueError(
            f"Configured dims exceed ES dense_vector limit: {expected_dims}"
        )
    if int(vectors.shape[1]) != expected_dims:
        raise ValueError(
            f"Embedding dims mismatch: expected={expected_dims}, actual={vectors.shape[1]}"
        )

    chunk_ids: list[str] = []
    with open(chunk_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            chunk_ids.append(data["chunk_id"])

    assert len(vectors) == len(chunk_ids), (
        f"Vector count ({len(vectors)}) != chunk_id count ({len(chunk_ids)})"
    )
    print(f"Loaded {len(vectors)} vectors (dims={vectors.shape[1]})")

    content_meta: dict[str, dict[str, Any]] = {}
    chunks = load_chunks_jsonl(chunks_path)
    for chunk in chunks:
        extra_meta = chunk.extra_meta if isinstance(chunk.extra_meta, dict) else {}
        content_meta[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_type": chunk.doc_type,
            "device_name": chunk.device_name,
            "equip_id": chunk.equip_id,
            "lang": chunk.lang,
            "tenant_id": str(extra_meta.get("tenant_id", "") or ""),
            "project_id": str(extra_meta.get("project_id", "") or ""),
            "chapter": chunk.chapter,
            "content_hash": chunk.content_hash,
        }

    missing_meta = [cid for cid in chunk_ids if cid not in content_meta]
    if missing_meta:
        preview = ", ".join(missing_meta[:5])
        raise ValueError(
            f"Missing metadata for {len(missing_meta)} chunk_ids from --chunks: {preview}"
        )

    body = {
        "settings": _build_chunk_v3_settings(),
        "mappings": get_chunk_v3_embed_mapping(
            dims=cfg["dims"],
            model_meta={
                "embedding_model": cfg["hf_name"],
                "dims": cfg["dims"],
                "normalize": cfg["normalize"],
                "query_prefix": cfg["query_prefix"],
                "document_prefix": cfg["document_prefix"],
            },
        ),
    }
    _prepare_index(es, index_name, body=body, recreate=recreate)

    def _gen_actions():
        for cid, vec in zip(chunk_ids, vectors):
            meta = content_meta[cid]
            yield {
                "_index": index_name,
                "_id": cid,
                "_source": {
                    "chunk_id": cid,
                    "doc_id": str(meta.get("doc_id", "") or ""),
                    "content_hash": str(meta.get("content_hash", "") or ""),
                    "doc_type": str(meta.get("doc_type", "") or ""),
                    "device_name": str(meta.get("device_name", "") or ""),
                    "equip_id": str(meta.get("equip_id", "") or ""),
                    "lang": str(meta.get("lang", "") or ""),
                    "tenant_id": str(meta.get("tenant_id", "") or ""),
                    "project_id": str(meta.get("project_id", "") or ""),
                    "chapter": str(meta.get("chapter", "") or ""),
                    "embedding": vec.tolist(),
                },
            }

    _set_refresh_interval(es, index_name, "-1")
    try:
        success, failed, sampled_errors = _run_streaming_bulk(
            es, _gen_actions(), chunk_size=batch_size
        )
    finally:
        _set_refresh_interval(es, index_name, "1s")

    print(f"Indexed {success} vectors to {index_name}")
    if failed:
        print(f"  Failed: {failed}")
        for err in sampled_errors:
            print(f"    {err}")


def verify_sync(
    model_key: str,
    content_index: str = CONTENT_INDEX,
    embed_index: str | None = None,
    output_json: str | None = None,
) -> bool:
    """content와 embed 인덱스 동기화 검증.

    Args:
        model_key: 모델 키

    Returns:
        True if synced
    """
    embed_index_name = embed_index or build_embed_index_name(model_key)
    es = _get_es_client()
    report: dict[str, Any] = {
        "model": model_key,
        "content_index": content_index,
        "embed_index": embed_index_name,
    }

    content_count = es.count(index=content_index)["count"]
    embed_count = es.count(index=embed_index_name)["count"]
    report["content_count"] = content_count
    report["embed_count"] = embed_count

    print(f"\nSync verification: {content_index} vs {embed_index_name}")
    print(f"  Content count: {content_count}")
    print(f"  Embed count:   {embed_count}")

    if content_count != embed_count:
        print(f"  FAIL: Document count mismatch ({content_count} vs {embed_count})")
        report["ok"] = False
        report["reason"] = "count_mismatch"
        _write_json(output_json, report)
        return False

    from elasticsearch.helpers import scan

    content_ids: set[str] = set()
    for doc in scan(
        es,
        index=content_index,
        query={"query": {"match_all": {}}},
        _source=["chunk_id"],
    ):
        content_ids.add(doc["_source"]["chunk_id"])

    embed_ids: set[str] = set()
    for doc in scan(
        es,
        index=embed_index_name,
        query={"query": {"match_all": {}}},
        _source=["chunk_id"],
    ):
        embed_ids.add(doc["_source"]["chunk_id"])

    missing_in_embed = content_ids - embed_ids
    missing_in_content = embed_ids - content_ids
    report["missing_in_embed"] = len(missing_in_embed)
    report["missing_in_content"] = len(missing_in_content)

    if missing_in_embed:
        print(f"  FAIL: {len(missing_in_embed)} chunk_ids in content but not in embed")
        for cid in list(missing_in_embed)[:5]:
            print(f"    - {cid}")
        report["ok"] = False
        report["reason"] = "missing_in_embed"
        _write_json(output_json, report)
        return False

    if missing_in_content:
        print(
            f"  FAIL: {len(missing_in_content)} chunk_ids in embed but not in content"
        )
        for cid in list(missing_in_content)[:5]:
            print(f"    - {cid}")
        report["ok"] = False
        report["reason"] = "missing_in_content"
        _write_json(output_json, report)
        return False

    overlap_ids = sorted(content_ids & embed_ids)
    sample_ids = overlap_ids[:200]
    report["sampled_hash_ids"] = len(sample_ids)
    if sample_ids:
        content_resp = es.mget(index=content_index, body={"ids": sample_ids})
        embed_resp = es.mget(index=embed_index_name, body={"ids": sample_ids})
        content_hash_map: dict[str, str] = {}
        embed_hash_map: dict[str, str] = {}

        for doc in content_resp.get("docs", []):
            if doc.get("found"):
                cid = str(doc.get("_id", ""))
                src = doc.get("_source", {})
                if cid:
                    content_hash_map[cid] = str(src.get("content_hash", ""))

        for doc in embed_resp.get("docs", []):
            if doc.get("found"):
                cid = str(doc.get("_id", ""))
                src = doc.get("_source", {})
                if cid:
                    embed_hash_map[cid] = str(src.get("content_hash", ""))

        mismatch_ids = [
            cid
            for cid in sample_ids
            if content_hash_map.get(cid, "") != embed_hash_map.get(cid, "")
        ]
        if mismatch_ids:
            print(
                f"  FAIL: content_hash mismatch on {len(mismatch_ids)} sampled chunk_ids"
            )
            for cid in mismatch_ids[:5]:
                print(
                    f"    - {cid}: content={content_hash_map.get(cid, '')} "
                    f"embed={embed_hash_map.get(cid, '')}"
                )
            report["ok"] = False
            report["reason"] = "content_hash_mismatch"
            report["content_hash_mismatch_count"] = len(mismatch_ids)
            _write_json(output_json, report)
            return False

    print("  PASS: All chunk_ids match")
    report["ok"] = True
    report["reason"] = "ok"
    _write_json(output_json, report)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: ES 적재 + 검증")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # content subcommand
    content_parser = subparsers.add_parser("content", help="Content 인덱스 적재")
    content_parser.add_argument("--chunks", required=True, help="JSONL 파일 경로")
    content_parser.add_argument("--content-index", default=CONTENT_INDEX)
    content_parser.add_argument("--recreate", action="store_true")
    content_parser.add_argument("--batch-size", type=int, default=500)

    # embed subcommand
    embed_parser = subparsers.add_parser("embed", help="Embed 인덱스 적재")
    embed_parser.add_argument(
        "--model", required=True, help="모델 키 (bge_m3, koe5, ...)"
    )
    embed_parser.add_argument("--embeddings", required=True, help=".npy 파일 경로")
    embed_parser.add_argument("--chunk-ids", required=True, help="chunk_ids JSONL 경로")
    embed_parser.add_argument("--chunks", required=True, help="all_chunks.jsonl")
    embed_parser.add_argument("--embed-index", default=None)
    embed_parser.add_argument("--recreate", action="store_true")
    embed_parser.add_argument("--batch-size", type=int, default=500)

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="동기화 검증")
    verify_parser.add_argument("--model", required=True, help="모델 키")
    verify_parser.add_argument("--content-index", default=CONTENT_INDEX)
    verify_parser.add_argument("--embed-index", default=None)
    verify_parser.add_argument("--output-json", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "content":
        ingest_content(
            chunks_path=args.chunks,
            content_index=args.content_index,
            batch_size=args.batch_size,
            recreate=args.recreate,
        )

    elif args.command == "embed":
        ingest_embeddings(
            model_key=args.model,
            embeddings_path=args.embeddings,
            chunk_ids_path=args.chunk_ids,
            chunks_path=args.chunks,
            embed_index=args.embed_index,
            batch_size=args.batch_size,
            recreate=args.recreate,
        )

    elif args.command == "verify":
        output_json = args.output_json
        if not output_json:
            output_json = f"data/chunks_v3/verify_{args.model}.json"
        ok = verify_sync(
            model_key=args.model,
            content_index=args.content_index,
            embed_index=args.embed_index,
            output_json=output_json,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
