#!/usr/bin/env python3
"""Paper B stability evaluation — supports S0-S9 condition matrix.

Usage:
    # Run S0 (prod-like baseline) on synthetic benchmark
    python scripts/paper_b/run_paper_b_eval.py --condition S0 \
        --queries data/synth_benchmarks/stability_bench_v1/queries.jsonl

    # Run S1 (deterministic protocol) on real corpus
    python scripts/paper_b/run_paper_b_eval.py --condition S1 \
        --queries data/paper_a/eval/query_gold_master.jsonl

    # Run S6 (SMQ-CR) with cached variants
    python scripts/paper_b/run_paper_b_eval.py --condition S6 \
        --queries data/synth_benchmarks/stability_bench_v1/queries.jsonl \
        --variant-cache data/paper_b/cached_variants.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import cast


DEFAULT_API_BASE_URL = "http://localhost:8011"
DEFAULT_QUERIES_PATH = "data/synth_benchmarks/stability_bench_v1/queries.jsonl"
DEFAULT_OUT_DIR = ".sisyphus/evidence/paper-b"
DEFAULT_K = 10
DEFAULT_REPEATS = 10

# ── Condition definitions (S0-S9) ─────────────────────────────────────────

CONDITION_PRESETS: dict[str, dict[str, object]] = {
    # Block 1: Cause analysis
    "S0": {
        "description": "Prod-like: MQ ON, no shard preference, unstable tie-break",
        "deterministic": False,
        "skip_mq": False,
        "rerank_enabled": False,
        "steps": ["translate", "mq", "retrieve"],
        "query_transform": None,
        "group_fusion": None,
    },
    "S1": {
        "description": "Deterministic protocol: MQ OFF, stable routing, stable tie-break",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": None,
        "group_fusion": None,
    },
    "S2": {
        "description": "S1 + reindex sensitivity (run after index rebuild)",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": None,
        "group_fusion": None,
    },
    # Block 2: T2 elimination
    "S3": {
        "description": "S1 + query canonicalization",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": "canonicalize",
        "group_fusion": None,
    },
    "S4": {
        "description": "S1 + result intersection across paraphrase group",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": None,
        "group_fusion": "intersection",
    },
    "S5": {
        "description": "S1 + score averaging across paraphrase group",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": None,
        "group_fusion": "score_average",
    },
    "S6": {
        "description": "S1 + SMQ-CR (fixed-variant consensus, hierarchical RRF)",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": None,
        "group_fusion": "smq_cr",
    },
    # Block 3: Ablation
    "S7": {
        "description": "S1 + ANN num_candidates sweep",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": False,
        "steps": ["retrieve"],
        "query_transform": None,
        "group_fusion": None,
    },
    "S8": {
        "description": "S1 + reranker ON",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": True,
        "steps": ["retrieve", "rerank"],
        "query_transform": None,
        "group_fusion": None,
    },
    "S9": {
        "description": "S6 + reranker ON",
        "deterministic": True,
        "skip_mq": True,
        "rerank_enabled": True,
        "steps": ["retrieve", "rerank"],
        "query_transform": None,
        "group_fusion": "smq_cr",
    },
}


# ── Abbreviation dictionary for S3 canonicalization ───────────────────────

ABBR_EXPANSIONS: dict[str, str] = {
    "PM": "Preventive Maintenance",
    "BM": "Breakdown Maintenance",
    "SOP": "Standard Operating Procedure",
    "TSG": "Troubleshooting Guide",
    "PEMS": "Process Equipment Monitoring System",
    "MFC": "Mass Flow Controller",
    "RF": "Radio Frequency",
    "CVD": "Chemical Vapor Deposition",
    "CMP": "Chemical Mechanical Polishing",
    "APC": "Advanced Process Control",
    "ESC": "Electrostatic Chuck",
    "TC": "Thermocouple",
    "WFR": "Wafer",
}

MIXED_LANG_MAP: dict[str, str] = {
    "프로시저": "절차",
    "매뉴얼": "설명서",
    "셋업": "설정",
    "트러블슈팅": "문제해결",
    "파라미터": "매개변수",
    "디바이스": "장비",
    "모듈": "모듈",
    "체크": "점검",
    "에러": "오류",
}


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryRow:
    qid: str
    group_id: str
    canonical_query: str
    query: str
    expected_doc_ids: list[str]
    paraphrase_level: str
    tags: list[str]


@dataclass(frozen=True)
class DocResult:
    doc_id: str
    score: float
    chunk_id: str


@dataclass(frozen=True)
class RunResult:
    qid: str
    group_id: str
    condition: str
    repeat_index: int
    top_k_docs: list[DocResult]
    top_k_doc_ids: list[str]
    latency_ms: float
    run_id: str
    effective_config_hash: str
    trace_id: str
    warnings: list[str]
    boundary_margin: float


# ── Utility functions ─────────────────────────────────────────────────────

def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    return {str(k): v for k, v in raw.items()}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ranked = sorted(values)
    rank = min(max(int(math.ceil(0.95 * len(ranked))) - 1, 0), len(ranked) - 1)
    return ranked[rank]


def _jaccard(left: list[str], right: list[str], k: int = DEFAULT_K) -> float:
    left_set = set(left[:k])
    right_set = set(right[:k])
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _exact_match(left: list[str], right: list[str], k: int = DEFAULT_K) -> float:
    return 1.0 if left[:k] == right[:k] else 0.0


def _hit_at_k(doc_ids: list[str], expected: list[str], k: int) -> float:
    gold = set(expected)
    return 1.0 if any(d in gold for d in doc_ids[:k]) else 0.0


def _mrr(doc_ids: list[str], expected: list[str], k: int) -> float:
    gold = set(expected)
    for i, d in enumerate(doc_ids[:k], start=1):
        if d in gold:
            return 1.0 / float(i)
    return 0.0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def _bootstrap_ci(
    values: list[float], n_samples: int = 10_000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) via percentile bootstrap."""
    import random
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    rng = random.Random(42)
    means: list[float] = []
    for _ in range(n_samples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, int(math.floor(alpha * n_samples)))
    hi_idx = min(n_samples - 1, int(math.ceil((1.0 - alpha) * n_samples)) - 1)
    return _mean(values), means[lo_idx], means[hi_idx]


# ── Query canonicalization (S3) ───────────────────────────────────────────

def _canonicalize_query(query: str) -> str:
    """Rule-based query canonicalization: abbreviation expansion + mixed-lang normalization."""
    result = query
    # Expand abbreviations (whole word match)
    for abbr, expansion in ABBR_EXPANSIONS.items():
        # Simple word-boundary replacement
        import re
        result = re.sub(rf'\b{re.escape(abbr)}\b', expansion, result)
    # Normalize mixed-language terms
    for foreign, native in MIXED_LANG_MAP.items():
        result = result.replace(foreign, native)
    # Remove filler words
    for filler in ["좀", "혹시", "혹시나", "제발", "please", "can you", "could you"]:
        result = result.replace(filler, "")
    # Normalize whitespace
    result = " ".join(result.split())
    return result


# ── HTTP helpers ──────────────────────────────────────────────────────────

def _build_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/api"):
        return f"{stripped}/retrieval/run"
    return f"{stripped}/api/retrieval/run"


def _build_payload(
    query: str,
    condition: dict[str, object],
    final_top_k: int = DEFAULT_K,
) -> dict[str, object]:
    return {
        "query": query,
        "steps": condition["steps"],
        "debug": False,
        "deterministic": condition["deterministic"],
        "final_top_k": final_top_k,
        "rerank_enabled": condition["rerank_enabled"],
        "auto_parse": False,
        "skip_mq": condition.get("skip_mq"),
        "device_names": None,
        "doc_types": None,
        "doc_types_strict": None,
        "equip_ids": None,
    }


def _post_json(
    endpoint: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> tuple[dict[str, object], float]:
    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with cast(
            http.client.HTTPResponse,
            urllib.request.urlopen(request, timeout=timeout_seconds),
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {endpoint}: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    try:
        parsed = cast(object, json.loads(raw.decode("utf-8")))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON response from retrieval endpoint") from exc
    mapped = _coerce_str_mapping(parsed)
    if mapped is None:
        raise RuntimeError("Unexpected response shape: JSON object expected")
    return mapped, elapsed_ms


# ── Response extraction ───────────────────────────────────────────────────

def _extract_docs(response: Mapping[str, object]) -> list[DocResult]:
    """Extract top-k documents with scores from API response."""
    docs_raw = response.get("docs")
    if not isinstance(docs_raw, list):
        return []
    results: list[DocResult] = []
    for item in cast(list[object], docs_raw):
        doc = _coerce_str_mapping(item)
        if doc is None:
            continue
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            continue
        score = 0.0
        score_raw = doc.get("score")
        if score_raw is not None:
            try:
                score = float(str(score_raw))
            except (ValueError, TypeError):
                pass
        metadata = _coerce_str_mapping(doc.get("metadata"))
        chunk_id = ""
        if metadata:
            chunk_id = str(metadata.get("chunk_id", "")).strip()
        results.append(DocResult(doc_id=doc_id, score=score, chunk_id=chunk_id))
    return results


def _compute_boundary_margin(docs: list[DocResult], k: int = DEFAULT_K) -> float:
    """BoundaryMargin@k = score(rank_k) - score(rank_{k+1})."""
    if len(docs) <= k:
        return 0.0
    return docs[k - 1].score - docs[k].score


def _extract_trace_id(response: Mapping[str, object]) -> str:
    trace = _coerce_str_mapping(response.get("trace"))
    if trace is None:
        return ""
    return str(trace.get("trace_id", "")).strip()


def _extract_warnings(response: Mapping[str, object]) -> list[str]:
    warnings_raw = response.get("warnings")
    if not isinstance(warnings_raw, list):
        return []
    return [str(item) for item in cast(list[object], warnings_raw)]


# ── Group-level fusion methods (S4, S5, S6) ──────────────────────────────

def _fusion_intersection(
    group_results: dict[str, list[DocResult]],
    k: int = DEFAULT_K,
) -> list[str]:
    """S4: Return documents appearing in ALL variants' top-K."""
    if not group_results:
        return []
    sets = [set(d.doc_id for d in docs[:k]) for docs in group_results.values()]
    common = sets[0]
    for s in sets[1:]:
        common &= s
    # Order by average rank across variants
    rank_sums: dict[str, float] = defaultdict(float)
    for docs in group_results.values():
        id_to_rank = {d.doc_id: i for i, d in enumerate(docs[:k])}
        for doc_id in common:
            rank_sums[doc_id] += id_to_rank.get(doc_id, k)
    return sorted(common, key=lambda d: rank_sums[d])[:k]


def _fusion_score_average(
    group_results: dict[str, list[DocResult]],
    k: int = DEFAULT_K,
) -> list[str]:
    """S5: Average scores across variants, re-rank."""
    if not group_results:
        return []
    score_sums: dict[str, float] = defaultdict(float)
    score_counts: dict[str, int] = defaultdict(int)
    n_variants = len(group_results)
    for docs in group_results.values():
        for doc in docs[:k]:
            score_sums[doc.doc_id] += doc.score
            score_counts[doc.doc_id] += 1
    # Docs not retrieved by a variant get score 0
    avg_scores = {
        doc_id: score_sums[doc_id] / n_variants
        for doc_id in score_sums
    }
    ranked = sorted(avg_scores.keys(), key=lambda d: -avg_scores[d])
    return ranked[:k]


def _fusion_smq_cr(
    variant_results: list[list[DocResult]],
    k: int = DEFAULT_K,
    rrf_k: int = 60,
) -> list[str]:
    """S6: Hierarchical RRF consensus over fixed variants."""
    if not variant_results:
        return []
    rrf_scores: dict[str, float] = defaultdict(float)
    for docs in variant_results:
        for rank, doc in enumerate(docs):
            rrf_scores[doc.doc_id] += 1.0 / (rrf_k + rank + 1)
    ranked = sorted(rrf_scores.keys(), key=lambda d: -rrf_scores[d])
    return ranked[:k]


# ── Query loading ─────────────────────────────────────────────────────────

def _load_queries(path: Path, limit: int | None) -> list[QueryRow]:
    rows: list[QueryRow] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            payload = line.split("|", 1)[-1].strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
            row = _coerce_str_mapping(parsed)
            if row is None:
                raise RuntimeError(f"JSON object expected at {path}:{line_no}")

            expected_raw = row.get("expected_doc_ids")
            if not isinstance(expected_raw, list):
                raise RuntimeError(
                    f"expected_doc_ids must be array at {path}:{line_no}"
                )
            expected_doc_ids = [
                str(item).strip()
                for item in cast(list[object], expected_raw)
                if str(item).strip()
            ]
            if not expected_doc_ids:
                raise RuntimeError(
                    f"expected_doc_ids must be non-empty at {path}:{line_no}"
                )

            tags_raw = row.get("tags")
            tags: list[str] = []
            if isinstance(tags_raw, list):
                tags = [str(t).strip() for t in cast(list[object], tags_raw) if str(t).strip()]

            rows.append(
                QueryRow(
                    qid=str(row.get("qid", "")).strip(),
                    group_id=str(row.get("group_id", "")).strip(),
                    canonical_query=str(row.get("canonical_query", "")).strip(),
                    query=str(row.get("query", "")).strip(),
                    expected_doc_ids=expected_doc_ids,
                    paraphrase_level=str(row.get("paraphrase_level", "")).strip(),
                    tags=tags,
                )
            )
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise RuntimeError(f"No queries loaded from {path}")
    return rows


def _load_variant_cache(path: Path | None) -> dict[str, list[str]]:
    """Load cached query variants for S6 (SMQ-CR).

    Format: JSONL with {"qid": "...", "variants": ["v1", "v2", ...]}
    Returns: {qid: [variant_query_1, variant_query_2, ...]}
    """
    if path is None or not path.exists():
        return {}
    cache: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("qid", ""))
            variants = obj.get("variants", [])
            if qid and variants:
                cache[qid] = [str(v) for v in variants]
    return cache


# ── Single-query retrieval ────────────────────────────────────────────────

def _run_single_query(
    endpoint: str,
    query: str,
    condition_config: dict[str, object],
    timeout_seconds: float,
    final_top_k: int = DEFAULT_K,
    fetch_extra: int = 0,
) -> tuple[list[DocResult], float, dict[str, object]]:
    """Run a single retrieval call. Returns (docs, latency_ms, raw_response)."""
    # Fetch extra docs for BoundaryMargin (need rank k+1)
    effective_k = final_top_k + fetch_extra
    payload = _build_payload(query, condition_config, final_top_k=effective_k)
    response, latency_ms = _post_json(endpoint, payload, timeout_seconds)
    docs = _extract_docs(response)
    return docs, latency_ms, response


# ── Main execution modes ──────────────────────────────────────────────────

def _run_standard(
    queries: list[QueryRow],
    endpoint: str,
    condition_id: str,
    condition_config: dict[str, object],
    repeats: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, object]], list[RunResult]]:
    """Standard mode: run each query N times, collect results.

    Used for S0, S1, S2, S3, S7, S8.
    """
    per_call_rows: list[dict[str, object]] = []
    all_results: list[RunResult] = []
    total = len(queries) * repeats
    count = 0

    for row in queries:
        # Apply query transform if needed
        query_text = row.query
        if condition_config.get("query_transform") == "canonicalize":
            query_text = _canonicalize_query(row.query)

        for repeat_idx in range(repeats):
            count += 1
            if count % 50 == 0 or count == total:
                print(f"  [{condition_id}] {count}/{total} queries processed", flush=True)

            docs, latency_ms, response = _run_single_query(
                endpoint, query_text, condition_config, timeout_seconds,
                final_top_k=DEFAULT_K, fetch_extra=1,
            )

            top_k_doc_ids = [d.doc_id for d in docs[:DEFAULT_K]]
            boundary_margin = _compute_boundary_margin(docs, DEFAULT_K)

            result = RunResult(
                qid=row.qid,
                group_id=row.group_id,
                condition=condition_id,
                repeat_index=repeat_idx,
                top_k_docs=docs[:DEFAULT_K],
                top_k_doc_ids=top_k_doc_ids,
                latency_ms=latency_ms,
                run_id=str(response.get("run_id", "")),
                effective_config_hash=str(response.get("effective_config_hash", "")),
                trace_id=_extract_trace_id(response),
                warnings=_extract_warnings(response),
                boundary_margin=boundary_margin,
            )
            all_results.append(result)

            per_call_rows.append({
                "qid": row.qid,
                "group_id": row.group_id,
                "canonical_query": row.canonical_query,
                "query": row.query,
                "query_sent": query_text,
                "paraphrase_level": row.paraphrase_level,
                "expected_doc_ids": row.expected_doc_ids,
                "tags": row.tags,
                "condition": condition_id,
                "repeat_index": repeat_idx,
                "latency_ms": latency_ms,
                "run_id": result.run_id,
                "effective_config_hash": result.effective_config_hash,
                "trace_id": result.trace_id,
                "warnings": result.warnings,
                "top_k_doc_ids": top_k_doc_ids,
                "top_k_scores": [d.score for d in docs[:DEFAULT_K]],
                "boundary_margin": boundary_margin,
            })

    return per_call_rows, all_results


def _run_group_fusion(
    queries: list[QueryRow],
    endpoint: str,
    condition_id: str,
    condition_config: dict[str, object],
    timeout_seconds: float,
    variant_cache: dict[str, list[str]],
) -> tuple[list[dict[str, object]], list[RunResult]]:
    """Group fusion mode: for each paraphrase group, fuse results.

    Used for S4 (intersection), S5 (score averaging), S6/S9 (SMQ-CR).
    """
    fusion_type = str(condition_config.get("group_fusion", ""))
    per_call_rows: list[dict[str, object]] = []
    all_results: list[RunResult] = []

    # Group queries by group_id
    groups: dict[str, list[QueryRow]] = defaultdict(list)
    for row in queries:
        groups[row.group_id].append(row)

    group_count = 0
    total_groups = len(groups)

    for group_id, group_queries in groups.items():
        group_count += 1
        if group_count % 10 == 0 or group_count == total_groups:
            print(f"  [{condition_id}] {group_count}/{total_groups} groups processed", flush=True)

        if fusion_type == "smq_cr":
            # S6/S9: For each query in group, generate consensus from cached variants
            for row in group_queries:
                variants = variant_cache.get(row.qid, [])
                if not variants:
                    # Fallback: use group queries as variants
                    variants = [q.query for q in group_queries if q.qid != row.qid]

                # Retrieve for original + variants
                variant_docs: list[list[DocResult]] = []
                total_latency = 0.0

                # Original query
                docs, lat, resp = _run_single_query(
                    endpoint, row.query, condition_config, timeout_seconds,
                    final_top_k=DEFAULT_K * 2,
                )
                variant_docs.append(docs)
                total_latency += lat

                for vq in variants[:3]:  # Cap at 3 variants (+ original = 4)
                    docs_v, lat_v, _ = _run_single_query(
                        endpoint, vq, condition_config, timeout_seconds,
                        final_top_k=DEFAULT_K * 2,
                    )
                    variant_docs.append(docs_v)
                    total_latency += lat_v

                # Fuse
                fused_ids = _fusion_smq_cr(variant_docs, DEFAULT_K)
                boundary_margin = 0.0  # Not directly computable for fused results

                result = RunResult(
                    qid=row.qid,
                    group_id=row.group_id,
                    condition=condition_id,
                    repeat_index=0,
                    top_k_docs=[],
                    top_k_doc_ids=fused_ids,
                    latency_ms=total_latency,
                    run_id="",
                    effective_config_hash="",
                    trace_id="",
                    warnings=[],
                    boundary_margin=boundary_margin,
                )
                all_results.append(result)
                per_call_rows.append({
                    "qid": row.qid,
                    "group_id": row.group_id,
                    "canonical_query": row.canonical_query,
                    "query": row.query,
                    "condition": condition_id,
                    "repeat_index": 0,
                    "latency_ms": total_latency,
                    "top_k_doc_ids": fused_ids,
                    "n_variants": len(variant_docs),
                    "fusion_type": fusion_type,
                    "expected_doc_ids": row.expected_doc_ids,
                    "tags": row.tags,
                    "boundary_margin": boundary_margin,
                })

        elif fusion_type in ("intersection", "score_average"):
            # S4/S5: Retrieve for each variant in group, then fuse
            group_doc_results: dict[str, list[DocResult]] = {}
            total_latency = 0.0

            for row in group_queries:
                docs, lat, _ = _run_single_query(
                    endpoint, row.query, condition_config, timeout_seconds,
                    final_top_k=DEFAULT_K * 2,
                )
                group_doc_results[row.qid] = docs
                total_latency += lat

            # Compute fused result for the group
            if fusion_type == "intersection":
                fused_ids = _fusion_intersection(group_doc_results, DEFAULT_K)
            else:
                fused_ids = _fusion_score_average(group_doc_results, DEFAULT_K)

            # Record result for each query in the group (same fused result)
            for row in group_queries:
                result = RunResult(
                    qid=row.qid,
                    group_id=row.group_id,
                    condition=condition_id,
                    repeat_index=0,
                    top_k_docs=[],
                    top_k_doc_ids=fused_ids,
                    latency_ms=total_latency / len(group_queries),
                    run_id="",
                    effective_config_hash="",
                    trace_id="",
                    warnings=[],
                    boundary_margin=0.0,
                )
                all_results.append(result)
                per_call_rows.append({
                    "qid": row.qid,
                    "group_id": row.group_id,
                    "canonical_query": row.canonical_query,
                    "query": row.query,
                    "condition": condition_id,
                    "repeat_index": 0,
                    "latency_ms": total_latency / len(group_queries),
                    "top_k_doc_ids": fused_ids,
                    "fusion_type": fusion_type,
                    "expected_doc_ids": row.expected_doc_ids,
                    "tags": row.tags,
                    "boundary_margin": 0.0,
                })

    return per_call_rows, all_results


# ── Metrics computation ───────────────────────────────────────────────────

def _compute_metrics(
    queries: list[QueryRow],
    all_results: list[RunResult],
    condition_id: str,
    repeats: int,
) -> dict[str, object]:
    """Compute all Paper B metrics from run results."""
    query_by_qid = {row.qid: row for row in queries}

    # Index results
    runs_by_qid: dict[str, list[list[str]]] = defaultdict(list)
    first_run_by_qid: dict[str, list[str]] = {}
    margins_by_qid: dict[str, list[float]] = defaultdict(list)

    for result in all_results:
        runs_by_qid[result.qid].append(result.top_k_doc_ids)
        if result.repeat_index == 0:
            first_run_by_qid[result.qid] = result.top_k_doc_ids
        margins_by_qid[result.qid].append(result.boundary_margin)

    # Effectiveness (from first run)
    hit5_vals: list[float] = []
    hit10_vals: list[float] = []
    mrr_vals: list[float] = []
    for qid, top10 in first_run_by_qid.items():
        q = query_by_qid.get(qid)
        if q is None:
            continue
        hit5_vals.append(_hit_at_k(top10, q.expected_doc_ids, 5))
        hit10_vals.append(_hit_at_k(top10, q.expected_doc_ids, 10))
        mrr_vals.append(_mrr(top10, q.expected_doc_ids, 10))

    # Repeat stability (T1)
    repeat_jaccard_per_q: list[float] = []
    repeat_exact_per_q: list[float] = []
    for qid, doc_lists in runs_by_qid.items():
        if len(doc_lists) <= 1:
            continue
        pj: list[float] = []
        pe: list[float] = []
        for left, right in combinations(doc_lists, 2):
            pj.append(_jaccard(left, right))
            pe.append(_exact_match(left, right))
        repeat_jaccard_per_q.append(_mean(pj))
        repeat_exact_per_q.append(_mean(pe))

    # Paraphrase stability (T2)
    first_run_by_group: dict[str, list[list[str]]] = defaultdict(list)
    for row in queries:
        first = first_run_by_qid.get(row.qid)
        if first is not None:
            first_run_by_group[row.group_id].append(first)

    para_jaccard_per_g: list[float] = []
    para_exact_per_g: list[float] = []
    for doc_lists in first_run_by_group.values():
        if len(doc_lists) <= 1:
            continue
        gj: list[float] = []
        ge: list[float] = []
        for left, right in combinations(doc_lists, 2):
            gj.append(_jaccard(left, right))
            ge.append(_exact_match(left, right))
        para_jaccard_per_g.append(_mean(gj))
        para_exact_per_g.append(_mean(ge))

    # Latency
    latencies = [r.latency_ms for r in all_results]

    # BoundaryMargin diagnostics (from first run)
    margins = [margins_by_qid[qid][0] for qid in first_run_by_qid if margins_by_qid[qid]]

    # Bootstrap CIs for primary metrics
    hit10_mean, hit10_lo, hit10_hi = _bootstrap_ci(hit10_vals)
    rj_mean, rj_lo, rj_hi = _bootstrap_ci(repeat_jaccard_per_q)
    pj_mean, pj_lo, pj_hi = _bootstrap_ci(para_jaccard_per_g)

    return {
        "condition": condition_id,
        "description": CONDITION_PRESETS.get(condition_id, {}).get("description", ""),
        "query_count": len(first_run_by_qid),
        "group_count": len(first_run_by_group),
        "repeats": repeats,
        # Primary metrics
        "hit@5": _mean(hit5_vals),
        "hit@10": hit10_mean,
        "hit@10_ci": [hit10_lo, hit10_hi],
        "MRR": _mean(mrr_vals),
        "RepeatJaccard@10": rj_mean,
        "RepeatJaccard@10_ci": [rj_lo, rj_hi],
        "RepeatExactMatch@10": _mean(repeat_exact_per_q),
        "ParaphraseJaccard@10": pj_mean,
        "ParaphraseJaccard@10_ci": [pj_lo, pj_hi],
        "ParaphraseExactMatch@10": _mean(para_exact_per_g),
        "p95_latency_ms": _p95(latencies),
        # Diagnostic
        "BoundaryMargin@10_mean": _mean(margins),
        "BoundaryMargin@10_p25": sorted(margins)[len(margins) // 4] if margins else 0.0,
        "BoundaryMargin@10_median": sorted(margins)[len(margins) // 2] if margins else 0.0,
        # Per-query diagnostics
        "per_query_repeat_jaccard": {
            qid: _mean([_jaccard(l, r) for l, r in combinations(runs_by_qid[qid], 2)])
            for qid in runs_by_qid if len(runs_by_qid[qid]) > 1
        },
        "per_group_paraphrase_jaccard": {
            gid: _mean([_jaccard(l, r) for l, r in combinations(dls, 2)])
            for gid, dls in first_run_by_group.items() if len(dls) > 1
        },
        "per_query_boundary_margin": {
            qid: margins_by_qid[qid][0]
            for qid in first_run_by_qid if margins_by_qid[qid]
        },
        # Legacy aliases
        "hit_at_5": _mean(hit5_vals),
        "hit_at_10": hit10_mean,
        "mrr": _mean(mrr_vals),
        "repeat_stability_jaccard_at_10": rj_mean,
        "paraphrase_stability_jaccard_at_10": pj_mean,
    }


# ── Argument parsing ──────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper B stability evaluation (S0-S9 conditions)"
    )
    parser.add_argument(
        "--condition", required=True,
        choices=list(CONDITION_PRESETS.keys()),
        help="Experimental condition (S0-S9)",
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--queries", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variant-cache",
        default=None,
        help="Path to cached query variants JSONL (for S6/S9 SMQ-CR)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    condition_id = args.condition
    condition_config = CONDITION_PRESETS[condition_id]

    queries = _load_queries(Path(args.queries), args.limit)
    endpoint = _build_endpoint(args.api_base_url)
    out_dir = Path(args.out_dir) / condition_id

    print(f"Paper B Eval: condition={condition_id}")
    print(f"  {condition_config['description']}")
    print(f"  queries={len(queries)}, repeats={args.repeats}")
    print(f"  endpoint={endpoint}")
    print(f"  output={out_dir}")
    print()

    fusion_type = condition_config.get("group_fusion")

    if fusion_type is not None:
        # Group-level fusion mode (S4, S5, S6, S9)
        variant_cache: dict[str, list[str]] = {}
        if fusion_type == "smq_cr":
            cache_path = Path(args.variant_cache) if args.variant_cache else None
            variant_cache = _load_variant_cache(cache_path)
            if not variant_cache:
                print("  WARNING: No variant cache loaded for SMQ-CR.")
                print("  Falling back to group-internal variants.")
        per_call_rows, all_results = _run_group_fusion(
            queries, endpoint, condition_id, condition_config,
            args.timeout_seconds, variant_cache,
        )
        effective_repeats = 1  # Group fusion runs once per query
    else:
        # Standard mode (S0, S1, S2, S3, S7, S8)
        per_call_rows, all_results = _run_standard(
            queries, endpoint, condition_id, condition_config,
            args.repeats, args.timeout_seconds,
        )
        effective_repeats = args.repeats

    # Compute metrics
    metrics = _compute_metrics(queries, all_results, condition_id, effective_repeats)

    # Write outputs
    _write_jsonl(out_dir / "results.jsonl", per_call_rows)
    _write_json(out_dir / "metrics.json", metrics)

    # Write summary to stdout
    print()
    print(f"=== {condition_id} Results ===")
    print(f"  Hit@5:                  {metrics['hit@5']:.4f}")
    print(f"  Hit@10:                 {metrics['hit@10']:.4f}  [{metrics['hit@10_ci'][0]:.4f}, {metrics['hit@10_ci'][1]:.4f}]")
    print(f"  MRR:                    {metrics['MRR']:.4f}")
    print(f"  RepeatJaccard@10:       {metrics['RepeatJaccard@10']:.4f}  [{metrics['RepeatJaccard@10_ci'][0]:.4f}, {metrics['RepeatJaccard@10_ci'][1]:.4f}]")
    print(f"  RepeatExactMatch@10:    {metrics['RepeatExactMatch@10']:.4f}")
    print(f"  ParaphraseJaccard@10:   {metrics['ParaphraseJaccard@10']:.4f}  [{metrics['ParaphraseJaccard@10_ci'][0]:.4f}, {metrics['ParaphraseJaccard@10_ci'][1]:.4f}]")
    print(f"  ParaphraseExactMatch@10:{metrics['ParaphraseExactMatch@10']:.4f}")
    print(f"  p95 latency (ms):       {metrics['p95_latency_ms']:.1f}")
    print(f"  BoundaryMargin@10 mean: {metrics['BoundaryMargin@10_mean']:.6f}")
    print()
    print(f"  Wrote: {out_dir / 'results.jsonl'}")
    print(f"  Wrote: {out_dir / 'metrics.json'}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
