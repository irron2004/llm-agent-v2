"""Phase 3: Retrieve top-k docs for dev queries and pool for judging.

Runs B0-B4.5, P1 on selected dev queries, saves top_doc_ids
regardless of gold labels (unlike evaluate_paper_a_master.py which skips no-gold queries).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (
    CrossEncoderReranker,
)
from backend.services.embedding_service import EmbeddingService
from elasticsearch import Elasticsearch

from scripts.paper_a._io import read_jsonl
from scripts.paper_a.canonicalize import compact_key


TOP_K = 10
ES_INDEX = "chunk_v3_embed_bge_m3_v1"
SYSTEMS = ["B0", "B1", "B2", "B3", "B4", "B4.5", "P1"]


def load_queries(eval_path: Path) -> list[dict]:
    return list(read_jsonl(eval_path))


def load_scope_data(doc_scope_path: Path, family_map_path: Path, shared_path: Path | None):
    doc_scope = {}
    for row in read_jsonl(doc_scope_path):
        doc_scope[row["doc_id"]] = row

    with open(family_map_path) as f:
        family_map = json.load(f)

    shared_doc_ids = set()
    if shared_path and shared_path.exists():
        shared_doc_ids = {l.strip() for l in open(shared_path) if l.strip()}

    return doc_scope, family_map, shared_doc_ids


def init_retrieval():
    es = Elasticsearch(search_settings.es_host)
    if not es.ping():
        raise RuntimeError(f"ES not reachable at {search_settings.es_host}")

    embedder = EmbeddingService()
    engine = EsSearchEngine(es_client=es)
    reranker = CrossEncoderReranker()
    return es, embedder, engine, reranker


def retrieve_bm25(engine, query: str, es_index: str, top_k: int, pre_filter=None):
    return engine.bm25_search(
        query=query, index=es_index, top_k=top_k, pre_filter=pre_filter
    )


def retrieve_dense(engine, embedder, query: str, es_index: str, top_k: int, pre_filter=None):
    vec = embedder.embed_query(query)
    return engine.dense_search(
        query_vector=vec, index=es_index, top_k=top_k, pre_filter=pre_filter
    )


def retrieve_hybrid(engine, embedder, query: str, es_index: str, top_k: int, pre_filter=None):
    vec = embedder.embed_query(query)
    return engine.hybrid_search(
        query=query, query_vector=vec, index=es_index, top_k=top_k, pre_filter=pre_filter
    )


def apply_rerank(reranker, query: str, results: list[dict], top_k: int):
    if not results:
        return results
    texts = [r.get("text", r.get("content", "")) for r in results]
    scores = reranker.rerank(query, texts)
    for r, s in zip(results, scores):
        r["rerank_score"] = s
    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results[:top_k]


def build_device_filter(query: dict, doc_scope: dict, family_map: dict, shared_doc_ids: set, system: str):
    """Build ES pre-filter for device-scoped systems (B4, B4.5, P1)."""
    allowed = query.get("allowed_devices", [])
    if not allowed:
        # Try to parse from question
        return None

    device_norm = allowed[0] if allowed else ""
    if not device_norm:
        return None

    # For B4: hard device filter
    # For B4.5: device + shared
    # For P1: device + shared + scope policy
    filter_doc_ids = set()
    for doc_id, scope in doc_scope.items():
        doc_device = scope.get("device_name", "")
        doc_device_norm = compact_key(doc_device) if doc_device else ""
        q_device_norm = compact_key(device_norm)

        if doc_device_norm == q_device_norm:
            filter_doc_ids.add(doc_id)
        elif system in ("B4.5", "P1") and doc_id in shared_doc_ids:
            filter_doc_ids.add(doc_id)

    if not filter_doc_ids:
        return None

    return {"terms": {"metadata.doc_id.keyword": list(filter_doc_ids)}}


def run_retrieval(query: dict, system: str, engine, embedder, reranker,
                  doc_scope, family_map, shared_doc_ids) -> list[str]:
    """Run a single system on a single query, return list of doc_ids."""
    q_text = query.get("question", "")
    pre_filter = None

    if system in ("B4", "B4.5", "P1"):
        pre_filter = build_device_filter(query, doc_scope, family_map, shared_doc_ids, system)
        if pre_filter is None and system in ("B4", "P1"):
            return []  # Can't filter without device

    try:
        if system == "B0":
            results = retrieve_bm25(engine, q_text, ES_INDEX, TOP_K)
        elif system == "B1":
            results = retrieve_dense(engine, embedder, q_text, ES_INDEX, TOP_K)
        elif system in ("B2", "B3", "B4", "B4.5", "P1"):
            results = retrieve_hybrid(engine, embedder, q_text, ES_INDEX, TOP_K, pre_filter=pre_filter)
            if system in ("B3", "B4", "B4.5", "P1"):
                results = apply_rerank(reranker, q_text, results, TOP_K)
        else:
            return []
    except Exception as e:
        print(f"  ERROR {system} {query['q_id']}: {e}")
        return []

    # Extract doc_ids (deduplicated, preserving order)
    seen = set()
    doc_ids = []
    for r in results:
        did = r.get("doc_id", r.get("metadata", {}).get("doc_id", ""))
        if did and did not in seen:
            seen.add(did)
            doc_ids.append(did)
    return doc_ids[:TOP_K]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--doc-scope", required=True)
    parser.add_argument("--family-map", required=True)
    parser.add_argument("--shared-doc-ids", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    queries = load_queries(Path(args.eval_set))
    print(f"Loaded {len(queries)} queries")

    doc_scope, family_map, shared_doc_ids = load_scope_data(
        Path(args.doc_scope), Path(args.family_map),
        Path(args.shared_doc_ids) if args.shared_doc_ids else None
    )

    es, embedder, engine, reranker = init_retrieval()
    print(f"ES connected, index={ES_INDEX}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run all systems on all queries
    all_results = []  # list of {q_id, system, doc_ids}
    pooled = defaultdict(set)  # q_id -> set of doc_ids

    for i, q in enumerate(queries):
        q_id = q["q_id"]
        print(f"[{i+1}/{len(queries)}] {q_id}: {q['question'][:60]}...")

        for sys_id in SYSTEMS:
            t0 = time.time()
            doc_ids = run_retrieval(q, sys_id, engine, embedder, reranker,
                                   doc_scope, family_map, shared_doc_ids)
            latency = (time.time() - t0) * 1000

            all_results.append({
                "q_id": q_id,
                "system": sys_id,
                "doc_ids": doc_ids,
                "latency_ms": round(latency, 1),
            })
            for did in doc_ids:
                pooled[q_id].add(did)

        n_pooled = len(pooled[q_id])
        print(f"  → pooled {n_pooled} unique docs from {len(SYSTEMS)} systems")

    # Save per-query retrieval results
    with open(out_dir / "retrieval_results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)

    # Save pooled (q_id, doc_id) pairs for judging
    pooled_pairs = []
    for q_id, doc_ids in pooled.items():
        q_data = next(q for q in queries if q["q_id"] == q_id)
        for did in sorted(doc_ids):
            pooled_pairs.append({
                "q_id": q_id,
                "question": q_data["question"],
                "device": q_data.get("canonical_device_name", ""),
                "scope_obs": q_data.get("scope_observability", ""),
                "doc_id": did,
            })

    with open(out_dir / "pooled_pairs.json", "w") as f:
        json.dump(pooled_pairs, f, ensure_ascii=False, indent=1)

    # Summary
    total_pairs = len(pooled_pairs)
    unique_docs = len(set(p["doc_id"] for p in pooled_pairs))
    print(f"\n=== Summary ===")
    print(f"Queries: {len(queries)}")
    print(f"Total pooled (q,doc) pairs: {total_pairs}")
    print(f"Unique documents: {unique_docs}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
