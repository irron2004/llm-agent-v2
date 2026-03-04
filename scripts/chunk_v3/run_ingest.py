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
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.common import load_chunks_jsonl
from scripts.chunk_v3.run_embedding import MODEL_CONFIGS

CONTENT_INDEX = "chunk_v3_content"


def _get_es_client():
    """Elasticsearch 클라이언트 생성."""
    from backend.config.settings import search_settings
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = {"hosts": [search_settings.es_host]}
    if search_settings.es_user and search_settings.es_password:
        kwargs["basic_auth"] = (search_settings.es_user, search_settings.es_password)
    kwargs["verify_certs"] = search_settings.es_verify_certs

    return Elasticsearch(**kwargs)


def _get_manager():
    """EsIndexManager 인스턴스 생성."""
    from backend.config.settings import search_settings
    from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager

    return EsIndexManager(
        es_host=search_settings.es_host,
        env=search_settings.es_env,
        es_user=search_settings.es_user if hasattr(search_settings, 'es_user') else None,
        es_password=search_settings.es_password if hasattr(search_settings, 'es_password') else None,
    )


def ingest_content(chunks_path: str, batch_size: int = 500) -> None:
    """chunk_v3_content에 원문+메타 적재 (embedding 없음).

    Args:
        chunks_path: JSONL 파일 경로
        batch_size: bulk 배치 크기
    """
    from backend.llm_infrastructure.elasticsearch.mappings import (
        get_chunk_v3_content_mapping,
        get_index_settings,
    )

    es = _get_es_client()
    chunks = load_chunks_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Create index if not exists
    if not es.indices.exists(index=CONTENT_INDEX):
        body = {
            "settings": get_index_settings(),
            "mappings": get_chunk_v3_content_mapping(),
        }
        es.indices.create(index=CONTENT_INDEX, body=body)
        print(f"Created index: {CONTENT_INDEX}")
    else:
        print(f"Index already exists: {CONTENT_INDEX}")

    # Bulk index
    from elasticsearch.helpers import bulk

    def _gen_actions():
        for chunk in chunks:
            doc = asdict(chunk)
            doc.pop("extra_meta", None)
            # Flatten extra_meta into top-level
            for k, v in (chunk.extra_meta or {}).items():
                doc[k] = v
            yield {
                "_index": CONTENT_INDEX,
                "_id": chunk.chunk_id,
                "_source": doc,
            }

    success, errors = bulk(es, _gen_actions(), chunk_size=batch_size, raise_on_error=False)
    print(f"Indexed {success} documents to {CONTENT_INDEX}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"    {err}")


def ingest_embeddings(
    model_key: str,
    embeddings_path: str,
    chunk_ids_path: str,
    batch_size: int = 500,
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
        get_index_settings,
    )

    cfg = MODEL_CONFIGS[model_key]
    index_name = f"chunk_v3_embed_{model_key}_v1"

    es = _get_es_client()

    # Load data
    vectors = np.load(embeddings_path)
    chunk_ids: list[str] = []
    with open(chunk_ids_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            chunk_ids.append(data["chunk_id"])

    assert len(vectors) == len(chunk_ids), (
        f"Vector count ({len(vectors)}) != chunk_id count ({len(chunk_ids)})"
    )
    print(f"Loaded {len(vectors)} vectors (dims={vectors.shape[1]})")

    # Load content chunks for metadata join
    # We need doc_type, device_name, chapter, content_hash from content index
    content_meta: dict[str, dict] = {}
    if es.indices.exists(index=CONTENT_INDEX):
        from elasticsearch.helpers import scan

        for doc in scan(es, index=CONTENT_INDEX, query={"query": {"match_all": {}}},
                        _source=["chunk_id", "doc_type", "device_name", "chapter", "content_hash"]):
            src = doc["_source"]
            content_meta[src["chunk_id"]] = src

    # Create embed index if not exists
    if not es.indices.exists(index=index_name):
        body = {
            "settings": get_index_settings(),
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
        es.indices.create(index=index_name, body=body)
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")

    # Bulk index
    from elasticsearch.helpers import bulk

    def _gen_actions():
        for i, (cid, vec) in enumerate(zip(chunk_ids, vectors)):
            meta = content_meta.get(cid, {})
            yield {
                "_index": index_name,
                "_id": cid,
                "_source": {
                    "chunk_id": cid,
                    "content_hash": meta.get("content_hash", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "device_name": meta.get("device_name", ""),
                    "chapter": meta.get("chapter", ""),
                    "embedding": vec.tolist(),
                },
            }

    success, errors = bulk(es, _gen_actions(), chunk_size=batch_size, raise_on_error=False)
    print(f"Indexed {success} vectors to {index_name}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"    {err}")


def verify_sync(model_key: str) -> bool:
    """content와 embed 인덱스 동기화 검증.

    Args:
        model_key: 모델 키

    Returns:
        True if synced
    """
    embed_index = f"chunk_v3_embed_{model_key}_v1"
    es = _get_es_client()

    # Document count
    content_count = es.count(index=CONTENT_INDEX)["count"]
    embed_count = es.count(index=embed_index)["count"]

    print(f"\nSync verification: {CONTENT_INDEX} vs {embed_index}")
    print(f"  Content count: {content_count}")
    print(f"  Embed count:   {embed_count}")

    if content_count != embed_count:
        print(f"  FAIL: Document count mismatch ({content_count} vs {embed_count})")
        return False

    # Chunk ID set comparison (sample-based for large indices)
    from elasticsearch.helpers import scan

    content_ids: set[str] = set()
    for doc in scan(es, index=CONTENT_INDEX, query={"query": {"match_all": {}}},
                    _source=["chunk_id"]):
        content_ids.add(doc["_source"]["chunk_id"])

    embed_ids: set[str] = set()
    for doc in scan(es, index=embed_index, query={"query": {"match_all": {}}},
                    _source=["chunk_id"]):
        embed_ids.add(doc["_source"]["chunk_id"])

    missing_in_embed = content_ids - embed_ids
    missing_in_content = embed_ids - content_ids

    if missing_in_embed:
        print(f"  FAIL: {len(missing_in_embed)} chunk_ids in content but not in embed")
        for cid in list(missing_in_embed)[:5]:
            print(f"    - {cid}")
        return False

    if missing_in_content:
        print(f"  FAIL: {len(missing_in_content)} chunk_ids in embed but not in content")
        for cid in list(missing_in_content)[:5]:
            print(f"    - {cid}")
        return False

    print("  PASS: All chunk_ids match")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: ES 적재 + 검증")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # content subcommand
    content_parser = subparsers.add_parser("content", help="Content 인덱스 적재")
    content_parser.add_argument("--chunks", required=True, help="JSONL 파일 경로")
    content_parser.add_argument("--batch-size", type=int, default=500)

    # embed subcommand
    embed_parser = subparsers.add_parser("embed", help="Embed 인덱스 적재")
    embed_parser.add_argument("--model", required=True, help="모델 키 (bge_m3, koe5, ...)")
    embed_parser.add_argument("--embeddings", required=True, help=".npy 파일 경로")
    embed_parser.add_argument("--chunk-ids", required=True, help="chunk_ids JSONL 경로")
    embed_parser.add_argument("--batch-size", type=int, default=500)

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="동기화 검증")
    verify_parser.add_argument("--model", required=True, help="모델 키")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "content":
        ingest_content(args.chunks, batch_size=args.batch_size)

    elif args.command == "embed":
        ingest_embeddings(
            model_key=args.model,
            embeddings_path=args.embeddings,
            chunk_ids_path=args.chunk_ids,
            batch_size=args.batch_size,
        )

    elif args.command == "verify":
        ok = verify_sync(args.model)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
