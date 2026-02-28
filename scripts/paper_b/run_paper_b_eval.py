#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import cast


DEFAULT_API_BASE_URL = "http://localhost:8011"
DEFAULT_QUERIES_PATH = "data/synth_benchmarks/stability_bench_v1/queries.jsonl"
DEFAULT_OUT_DIR = ".sisyphus/evidence/paper-b"
DEFAULT_K = 10
DEFAULT_REPEATS = 10


@dataclass(frozen=True)
class CliArgs:
    api_base_url: str
    queries: str
    out_dir: str
    repeats: int
    timeout_seconds: float
    mode_name: str
    deterministic: bool
    limit: int | None


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
class RunResult:
    qid: str
    group_id: str
    mode: str
    repeat_index: int
    top_k_doc_ids: list[str]
    latency_ms: float
    run_id: str
    effective_config_hash: str
    trace_id: str
    warnings: list[str]


def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    mapped: dict[str, object] = {}
    for key, item in raw.items():
        mapped[str(key)] = item
    return mapped


def _parse_bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value: {value}")


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Run Paper B stability evaluation")
    _ = parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    _ = parser.add_argument("--queries", default=DEFAULT_QUERIES_PATH)
    _ = parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    _ = parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    _ = parser.add_argument("--timeout-seconds", type=float, default=60.0)
    _ = parser.add_argument(
        "--mode-name",
        default="deterministic_protocol",
        help="Mode label written to each per-call row",
    )
    _ = parser.add_argument(
        "--deterministic",
        default="true",
        help="Deterministic flag for this run (true/false)",
    )
    _ = parser.add_argument("--limit", type=int, default=None)
    parsed = parser.parse_args()

    repeats = cast(int, parsed.repeats)
    timeout_seconds = cast(float, parsed.timeout_seconds)
    mode_name = str(cast(object, parsed.mode_name)).strip()
    deterministic = _parse_bool_text(str(cast(object, parsed.deterministic)))
    limit = cast(int | None, parsed.limit)

    if repeats <= 0:
        raise RuntimeError("--repeats must be >= 1")
    if timeout_seconds <= 0:
        raise RuntimeError("--timeout-seconds must be > 0")
    if not mode_name:
        raise RuntimeError("--mode-name must be non-empty")
    if limit is not None and limit <= 0:
        raise RuntimeError("--limit must be >= 1 when provided")

    return CliArgs(
        api_base_url=str(cast(object, parsed.api_base_url)),
        queries=str(cast(object, parsed.queries)),
        out_dir=str(cast(object, parsed.out_dir)),
        repeats=repeats,
        timeout_seconds=timeout_seconds,
        mode_name=mode_name,
        deterministic=deterministic,
        limit=limit,
    )


def _load_queries(path: Path, limit: int | None) -> list[QueryRow]:
    rows: list[QueryRow] = []
    try:
        with path.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                payload = line.split("|", 1)[-1].strip()
                if not payload:
                    continue
                try:
                    parsed = cast(object, json.loads(payload))
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
                expected_items = cast(list[object], expected_raw)
                expected_doc_ids = [
                    str(item).strip() for item in expected_items if str(item).strip()
                ]
                if not expected_doc_ids:
                    raise RuntimeError(
                        f"expected_doc_ids must be non-empty at {path}:{line_no}"
                    )

                tags_raw = row.get("tags")
                tags = []
                if isinstance(tags_raw, list):
                    tag_items = cast(list[object], tags_raw)
                    tags = [
                        str(item).strip() for item in tag_items if str(item).strip()
                    ]

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
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if not rows:
        raise RuntimeError(f"No queries loaded from {path}")

    return rows


def _build_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/api"):
        return f"{stripped}/retrieval/run"
    return f"{stripped}/api/retrieval/run"


def _build_payload(query: str, deterministic: bool) -> dict[str, object]:
    return {
        "query": query,
        "steps": ["retrieve"],
        "debug": False,
        "deterministic": deterministic,
        "final_top_k": DEFAULT_K,
        "rerank_enabled": False,
        "auto_parse": False,
        "skip_mq": None,
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


def _extract_top_k_doc_ids(response: Mapping[str, object]) -> list[str]:
    docs_raw = response.get("docs")
    if not isinstance(docs_raw, list):
        return []
    doc_ids: list[str] = []
    for item in cast(list[object], docs_raw):
        doc = _coerce_str_mapping(item)
        if doc is None:
            continue
        doc_id = str(doc.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.append(doc_id)
        if len(doc_ids) >= DEFAULT_K:
            break
    return doc_ids


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


def _hit_at_k(doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    expected = set(expected_doc_ids)
    return 1.0 if any(doc_id in expected for doc_id in doc_ids[:k]) else 0.0


def _mrr(doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    expected = set(expected_doc_ids)
    for index, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in expected:
            return 1.0 / float(index)
    return 0.0


def _jaccard_at_10(left: list[str], right: list[str]) -> float:
    left_set = set(left[:DEFAULT_K])
    right_set = set(right[:DEFAULT_K])
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _exact_match_at_10(left: list[str], right: list[str]) -> float:
    return 1.0 if left[:DEFAULT_K] == right[:DEFAULT_K] else 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ranked = sorted(values)
    rank = int(math.ceil(0.95 * len(ranked))) - 1
    rank = min(max(rank, 0), len(ranked) - 1)
    return ranked[rank]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            _ = fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        _ = fp.write("\n")


def main() -> int:
    args = _parse_args()
    queries = _load_queries(Path(args.queries), args.limit)
    endpoint = _build_endpoint(args.api_base_url)

    per_call_rows: list[dict[str, object]] = []
    all_run_results: list[RunResult] = []
    latencies_ms: list[float] = []

    for row in queries:
        for repeat_index in range(args.repeats):
            payload = _build_payload(row.query, deterministic=args.deterministic)
            response, latency_ms = _post_json(endpoint, payload, args.timeout_seconds)
            top_k = _extract_top_k_doc_ids(response)
            result = RunResult(
                qid=row.qid,
                group_id=row.group_id,
                mode=args.mode_name,
                repeat_index=repeat_index,
                top_k_doc_ids=top_k,
                latency_ms=latency_ms,
                run_id=str(response.get("run_id", "")),
                effective_config_hash=str(response.get("effective_config_hash", "")),
                trace_id=_extract_trace_id(response),
                warnings=_extract_warnings(response),
            )
            all_run_results.append(result)
            latencies_ms.append(latency_ms)

            per_call_rows.append(
                {
                    "qid": row.qid,
                    "group_id": row.group_id,
                    "canonical_query": row.canonical_query,
                    "query": row.query,
                    "paraphrase_level": row.paraphrase_level,
                    "expected_doc_ids": row.expected_doc_ids,
                    "tags": row.tags,
                    "mode": result.mode,
                    "repeat_index": result.repeat_index,
                    "request_payload": payload,
                    "latency_ms": result.latency_ms,
                    "run_id": result.run_id,
                    "effective_config_hash": result.effective_config_hash,
                    "trace_id": result.trace_id,
                    "warnings": result.warnings,
                    "top_k_doc_ids": result.top_k_doc_ids,
                    "top10_doc_ids": result.top_k_doc_ids,
                }
            )
    runs_by_qid: dict[str, list[list[str]]] = defaultdict(list)
    first_run_by_qid: dict[str, list[str]] = {}
    for result in all_run_results:
        runs_by_qid[result.qid].append(result.top_k_doc_ids)
        if result.repeat_index == 0:
            first_run_by_qid[result.qid] = result.top_k_doc_ids

    query_by_qid = {row.qid: row for row in queries}

    hit_at_5_values: list[float] = []
    hit_at_10_values: list[float] = []
    mrr_values: list[float] = []
    for qid, top10 in first_run_by_qid.items():
        query = query_by_qid[qid]
        hit_at_5_values.append(_hit_at_k(top10, query.expected_doc_ids, 5))
        hit_at_10_values.append(_hit_at_k(top10, query.expected_doc_ids, 10))
        mrr_values.append(_mrr(top10, query.expected_doc_ids, 10))

    repeat_jaccard_per_query: list[float] = []
    repeat_exact_per_query: list[float] = []
    for doc_lists in runs_by_qid.values():
        if len(doc_lists) <= 1:
            continue
        pair_jaccard: list[float] = []
        pair_exact: list[float] = []
        for left, right in combinations(doc_lists, 2):
            pair_jaccard.append(_jaccard_at_10(left, right))
            pair_exact.append(_exact_match_at_10(left, right))
        repeat_jaccard_per_query.append(_mean(pair_jaccard))
        repeat_exact_per_query.append(_mean(pair_exact))

    first_run_by_group: dict[str, list[list[str]]] = defaultdict(list)
    for row in queries:
        first = first_run_by_qid.get(row.qid)
        if first is not None:
            first_run_by_group[row.group_id].append(first)

    paraphrase_jaccard_per_group: list[float] = []
    paraphrase_exact_per_group: list[float] = []
    for doc_lists in first_run_by_group.values():
        if len(doc_lists) <= 1:
            continue
        group_pair_jaccard: list[float] = []
        group_pair_exact: list[float] = []
        for left, right in combinations(doc_lists, 2):
            group_pair_jaccard.append(_jaccard_at_10(left, right))
            group_pair_exact.append(_exact_match_at_10(left, right))
        paraphrase_jaccard_per_group.append(_mean(group_pair_jaccard))
        paraphrase_exact_per_group.append(_mean(group_pair_exact))

    hit_at_5 = _mean(hit_at_5_values)
    hit_at_10 = _mean(hit_at_10_values)
    mrr = _mean(mrr_values)
    repeat_jaccard = _mean(repeat_jaccard_per_query)
    repeat_exact = _mean(repeat_exact_per_query)
    paraphrase_jaccard = _mean(paraphrase_jaccard_per_group)
    paraphrase_exact = _mean(paraphrase_exact_per_group)

    # Calculate unique query count
    unique_query_count = len(first_run_by_qid)

    metrics: dict[str, object] = {
        # Paper-B spec keys
        "hit@5": hit_at_5,
        "hit@10": hit_at_10,
        "MRR": mrr,
        "RepeatJaccard@10": repeat_jaccard,
        "RepeatExactMatch@10": repeat_exact,
        "ParaphraseJaccard@10": paraphrase_jaccard,
        "ParaphraseExactMatch@10": paraphrase_exact,
        "p95_latency_ms": _p95(latencies_ms),
        # Legacy aliases for generate_paper_b_assets.py
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "mrr": mrr,
        "repeat_stability_jaccard_at_10": repeat_jaccard,
        "paraphrase_stability_jaccard_at_10": paraphrase_jaccard,
        "query_count": unique_query_count,
        "deterministic_repeats": args.repeats,
        "primary_mode_name": args.mode_name,
    }

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "results.jsonl", per_call_rows)
    _write_json(out_dir / "metrics.json", metrics)

    print(f"Wrote per-call results: {out_dir / 'results.jsonl'}")
    print(f"Wrote aggregate metrics: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
