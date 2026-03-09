"""Build Gold Doc Expansion Candidate Pack for Paper A.

Reads query_gold_master JSONL, finds queries with empty gold_doc_ids,
runs ES hybrid retrieval to fetch top-k candidate documents per query,
and outputs a JSON report + Markdown checklist for PE review.

Requires ES access (same infra as evaluate_paper_a_master.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (
    CrossEncoderReranker,
)
from backend.services.embedding_service import EmbeddingService
from elasticsearch import Elasticsearch

from scripts.paper_a._io import read_jsonl

# Priority order for scope_observability
_SCOPE_PRIORITY: dict[str, int] = {
    "explicit_device": 0,
    "implicit": 1,
    "explicit_equip": 2,
    "ambiguous": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build gold doc expansion candidate pack for Paper A PE review"
    )
    _ = parser.add_argument(
        "--eval-set",
        required=True,
        help="Path to query_gold_master JSONL",
    )
    _ = parser.add_argument(
        "--corpus-filter",
        required=True,
        help="Path to corpus_doc_ids.txt",
    )
    _ = parser.add_argument(
        "--out-json",
        required=True,
        help="Output JSON path",
    )
    _ = parser.add_argument(
        "--out-md",
        required=True,
        help="Output Markdown checklist path",
    )
    _ = parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ES candidate docs per query (default: 5)",
    )
    _ = parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of candidate queries per scope group to include (default: 30)",
    )
    _ = parser.add_argument(
        "--index",
        default=None,
        help="ES alias/index override",
    )
    _ = parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking to ES candidates",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            loaded = cast(object, json.loads(raw))
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"Invalid JSONL row at line {line_no}: expected object"
                )
            rows.append(cast(dict[str, object], loaded))
    return rows


def _scope_priority(scope: str) -> int:
    return _SCOPE_PRIORITY.get(scope, 99)


def _build_es_client() -> Elasticsearch:
    kwargs: dict[str, Any] = {"hosts": [search_settings.es_host], "verify_certs": True}
    if search_settings.es_user and search_settings.es_password:
        kwargs["basic_auth"] = (search_settings.es_user, search_settings.es_password)
    return Elasticsearch(**kwargs)


def _resolve_index(index_override: str | None) -> tuple[str, str]:
    """Resolve ES alias/index and return (alias, resolved_index)."""
    manager = EsIndexManager(
        es_host=search_settings.es_host, env=search_settings.es_env,
        index_prefix=search_settings.es_index_prefix,
        es_user=search_settings.es_user or None,
        es_password=search_settings.es_password or None, verify_certs=True,
    )
    alias = index_override or manager.get_alias_name()
    target = manager.get_alias_target()
    if not target:
        return alias, alias
    return alias, target


def _retrieve_candidates(
    *,
    question: str,
    es_engine: EsSearchEngine,
    embed_svc: EmbeddingService,
    base_filter: dict[str, Any],
    reranker: CrossEncoderReranker | None,
    top_k: int,
) -> list[dict[str, str | float]]:
    """Run hybrid search and return top-k candidate doc summaries."""
    vector = np.asarray(embed_svc.embed_query(question), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm

    hits = es_engine.hybrid_search(
        query_vector=list(vector), query_text=question,
        top_k=top_k, dense_weight=0.7, sparse_weight=0.3,
        filters=base_filter, use_rrf=True, rrf_k=60,
    )

    results = [
        RetrievalResult(
            doc_id=h.doc_id, content=h.content,
            score=h.score, metadata=h.metadata, raw_text=h.raw_text,
        )
        for h in hits
    ]

    if reranker is not None and results:
        results = reranker.rerank(query=question, results=results, top_k=top_k)

    # Deduplicate by doc_id
    seen: set[str] = set()
    candidates: list[dict[str, str | float]] = []
    for r in results:
        did = str(r.doc_id or "")
        if did in seen or not did:
            continue
        seen.add(did)
        meta = r.metadata or {}
        candidates.append({
            "doc_id": did,
            "score": round(float(r.score or 0.0), 4),
            "device_name": str(meta.get("device_name", "")),
            "doc_type": str(meta.get("doc_type", "")),
            "snippet": str(r.content or "")[:200],
        })
        if len(candidates) >= top_k:
            break

    return candidates


def _build_candidates(
    rows: list[dict[str, object]],
    top_n: int,
    seed: int | None,
) -> list[dict[str, object]]:
    """Extract empty-gold queries and return sorted + sampled candidates."""
    empty_gold: list[dict[str, object]] = []
    for row in rows:
        gold_doc_ids = row.get("gold_doc_ids")
        is_empty = (
            gold_doc_ids is None
            or (isinstance(gold_doc_ids, list) and len(gold_doc_ids) == 0)
        )
        if not is_empty:
            continue

        q_id = str(row.get("q_id") or "")
        question = str(row.get("question") or "")
        scope_observability = str(row.get("scope_observability") or "unknown")
        intent_primary = str(row.get("intent_primary") or "")

        allowed_devices_raw = row.get("allowed_devices")
        allowed_devices: list[str] = (
            [str(d) for d in cast(list[object], allowed_devices_raw)]
            if isinstance(allowed_devices_raw, list)
            else []
        )

        empty_gold.append(
            {
                "q_id": q_id,
                "question": question,
                "scope_observability": scope_observability,
                "intent_primary": intent_primary,
                "allowed_devices": allowed_devices,
                "status": "needs_gold_labeling",
            }
        )

    # Sort by priority then q_id for determinism
    empty_gold.sort(key=lambda r: (_scope_priority(str(r["scope_observability"])), str(r["q_id"])))

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Sample top_n per scope group
    by_scope: dict[str, list[dict[str, object]]] = {}
    for candidate in empty_gold:
        scope = str(candidate["scope_observability"])
        by_scope.setdefault(scope, []).append(candidate)

    sampled: list[dict[str, object]] = []
    for scope in sorted(by_scope.keys(), key=_scope_priority):
        group = by_scope[scope]
        if len(group) > top_n:
            group = rng.sample(group, top_n)
            group.sort(key=lambda r: str(r["q_id"]))
        sampled.extend(group)

    return sampled


def _write_json(
    path: Path,
    total_queries: int,
    empty_gold_count: int,
    candidates: list[dict[str, object]],
    generated_at: str,
    es_index: str,
    top_k: int,
    rerank: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "total_queries": total_queries,
        "empty_gold_count": empty_gold_count,
        "candidate_count": len(candidates),
        "es_index": es_index,
        "top_k_per_query": top_k,
        "rerank_applied": rerank,
        "candidates": candidates,
        "generated_at": generated_at,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_md(
    path: Path,
    total_queries: int,
    empty_gold_count: int,
    candidates: list[dict[str, object]],
    generated_at: str,
    es_index: str,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    pct = (empty_gold_count / total_queries * 100) if total_queries else 0.0

    lines: list[str] = [
        f"# Gold Doc Expansion Candidates ({generated_at})",
        "",
        "## Summary",
        f"- Total queries: {total_queries}",
        f"- Missing gold: {empty_gold_count} ({pct:.1f}%)",
        f"- Candidates in this pack: {len(candidates)}",
        f"- ES index: {es_index}",
        f"- Top-k per query: {top_k}",
        "",
        "## PE Review Instructions",
        "",
        "For each query below, review the ES candidate documents and mark",
        "which doc_ids are relevant (gold). Update `gold_doc_ids` in",
        "`query_gold_master_v0_5.jsonl` accordingly.",
        "",
        "## Candidates by Priority",
        "",
    ]

    # Group by scope in priority order
    by_scope: dict[str, list[dict[str, object]]] = {}
    for c in candidates:
        scope = str(c["scope_observability"])
        by_scope.setdefault(scope, []).append(c)

    scope_labels: dict[str, str] = {
        "explicit_device": "explicit_device (highest priority)",
        "implicit": "implicit",
        "explicit_equip": "explicit_equip",
        "ambiguous": "ambiguous",
    }

    for scope in sorted(by_scope.keys(), key=_scope_priority):
        group = by_scope[scope]
        label = scope_labels.get(scope, scope)
        lines.append(f"### {label}")
        lines.append("")
        for c in group:
            q_id = c["q_id"]
            question = c["question"]
            devices = cast(list[str], c["allowed_devices"])
            devices_str = ", ".join(devices) if devices else "(none)"
            lines.append(
                f"#### q_id: `{q_id}`"
            )
            lines.append(f"- **Question**: {question}")
            lines.append(f"- **Devices**: [{devices_str}]")
            lines.append(f"- **Scope**: {c['scope_observability']} | **Intent**: {c.get('intent_primary', '')}")
            es_docs = cast(list[dict[str, object]], c.get("es_candidates", []))
            if es_docs:
                lines.append(f"- **ES candidates** (top-{len(es_docs)}):")
                for i, doc in enumerate(es_docs, 1):
                    did = doc.get("doc_id", "")
                    score = doc.get("score", 0.0)
                    device = doc.get("device_name", "")
                    dtype = doc.get("doc_type", "")
                    snippet = str(doc.get("snippet", ""))[:120]
                    lines.append(
                        f"  - [ ] `{did}` (score={score}, device={device}, type={dtype})"
                    )
                    if snippet:
                        lines.append(f"    > {snippet}")
            else:
                lines.append("- **ES candidates**: (no retrieval run)")
            lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def run() -> int:
    args = parse_args()
    eval_set_path = Path(cast(str, args.eval_set))
    corpus_filter_path = Path(cast(str, args.corpus_filter))
    out_json_path = Path(cast(str, args.out_json))
    out_md_path = Path(cast(str, args.out_md))
    top_k = int(args.top_k)
    top_n = int(args.top_n)
    do_rerank = bool(args.rerank)
    seed: int | None = args.seed

    if not eval_set_path.exists():
        print(f"ERROR: eval-set file not found: {eval_set_path}", file=sys.stderr)
        return 1
    if not corpus_filter_path.exists():
        print(f"ERROR: corpus-filter file not found: {corpus_filter_path}", file=sys.stderr)
        return 1

    try:
        rows = _load_jsonl(eval_set_path)
    except Exception as exc:
        print(f"ERROR: failed to load eval-set: {exc}", file=sys.stderr)
        return 1

    total_queries = len(rows)
    empty_gold_count = sum(
        1 for r in rows
        if not r.get("gold_doc_ids")
    )

    candidates = _build_candidates(rows, top_n=top_n, seed=seed)
    generated_at = str(date.today())

    # --- ES retrieval for candidate documents ---
    print(f"Connecting to ES at {search_settings.es_host} ...")
    alias_name, resolved_index = _resolve_index(cast(str | None, args.index))
    print(f"Resolved index: {resolved_index}")

    corpus_doc_ids = [
        line.strip()
        for line in corpus_filter_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not corpus_doc_ids:
        print("ERROR: corpus filter is empty", file=sys.stderr)
        return 1

    es_engine = EsSearchEngine(
        es_client=_build_es_client(),
        index_name=resolved_index,
        text_fields=["search_text^1.0", "chunk_summary^0.7", "chunk_keywords^0.8"],
    )
    base_filter = es_engine.build_filter(doc_ids=corpus_doc_ids)
    if base_filter is None:
        print("ERROR: failed to build corpus filter", file=sys.stderr)
        return 1

    embed_svc = EmbeddingService(
        method=rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )

    reranker: CrossEncoderReranker | None = None
    if do_rerank:
        reranker = CrossEncoderReranker(device=rag_settings.embedding_device)
        print(f"Reranker: {reranker.model_name}")

    print(f"Fetching ES candidates for {len(candidates)} queries (top_k={top_k}) ...")
    t0 = time.monotonic()
    for i, cand in enumerate(candidates):
        question = str(cand["question"])
        es_docs = _retrieve_candidates(
            question=question,
            es_engine=es_engine,
            embed_svc=embed_svc,
            base_filter=cast(dict[str, Any], base_filter),
            reranker=reranker,
            top_k=top_k,
        )
        cand["es_candidates"] = es_docs
        if (i + 1) % 20 == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{len(candidates)}] {elapsed:.1f}s")

    elapsed = time.monotonic() - t0
    print(f"ES retrieval done in {elapsed:.1f}s")

    try:
        _write_json(
            out_json_path, total_queries, empty_gold_count, candidates,
            generated_at, resolved_index, top_k, do_rerank,
        )
        _write_md(
            out_md_path, total_queries, empty_gold_count, candidates,
            generated_at, resolved_index, top_k,
        )
    except Exception as exc:
        print(f"ERROR: failed to write outputs: {exc}", file=sys.stderr)
        return 1

    print(f"total_queries   : {total_queries}")
    print(f"empty_gold_count: {empty_gold_count} ({empty_gold_count/total_queries*100:.1f}%)")
    print(f"candidates       : {len(candidates)}")
    print(f"es_index         : {resolved_index}")
    print(f"out_json         : {out_json_path}")
    print(f"out_md           : {out_md_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
