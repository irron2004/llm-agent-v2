"""Device parser accuracy measurement for Paper A.

Measures how accurately the rule-based device parser (from langgraph_agent.py)
extracts device names from 578 evaluation queries, and compares oracle B4 vs
real-parser B4 retrieval performance.

Usage:
    cd /home/hskim/work/llm-agent-v2
    uv run python scripts/paper_a/measure_parser_accuracy.py

Outputs:
    data/paper_a/parser_accuracy_report.json
"""

from __future__ import annotations

import json
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a.canonicalize import compact_key
from scripts.paper_a._io import read_jsonl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_DOC_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
CORPUS_FILTER_PATH = ROOT / ".sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt"
FAMILY_MAP_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/family_map.json"
OUT_PATH = ROOT / "data/paper_a/parser_accuracy_report.json"
DIFF_CSV_PATH = ROOT / "data/paper_a/parser_accuracy_per_query_diff.csv"
EVIDENCE_MD_PATH = (
    ROOT / "docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md"
)

TOP_K = 10


# ---------------------------------------------------------------------------
# Device parser (mirrors _extract_devices_from_query from langgraph_agent.py)
# ---------------------------------------------------------------------------


def _is_valid_device_candidate(name: str) -> bool:
    cleaned = str(name).strip()
    if not cleaned:
        return False
    compact = compact_key(cleaned)
    if compact in {"all", "etc"}:
        return False
    token = "".join(ch for ch in cleaned if ch.isalnum())
    # Short pure-alpha tokens (APC, ALL etc.) are noise labels
    if token.isalpha() and len(token) <= 4:
        return False
    return True


def parse_devices_from_query(device_names: list[str], query: str) -> list[str]:
    """Rule-based device extraction — mirrors langgraph_agent._extract_devices_from_query."""
    if not device_names or not query:
        return []
    query_compact = compact_key(query)
    matches: list[str] = []
    for name in device_names:
        cleaned = str(name).strip()
        if not cleaned:
            continue
        if not _is_valid_device_candidate(cleaned):
            continue
        cand_compact = compact_key(cleaned)
        if cand_compact and cand_compact in query_compact:
            matches.append(cleaned)
    # STRICT: only one device (same policy as production)
    seen: set[str] = set()
    deduped: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped[:1]


def parse_equip_ids_from_query(equip_ids: set[str], query: str) -> list[str]:
    query_upper = str(query or "").upper()
    return [
        equip_id
        for equip_id in sorted(equip_ids)
        if equip_id and equip_id in query_upper
    ][:1]


# ---------------------------------------------------------------------------
# Load policy artifacts
# ---------------------------------------------------------------------------


def _load_doc_scope(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_family_map(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    device_to_family: dict[str, str] = data.get("device_to_family", {})
    families: dict[str, list[str]] = {
        fid: [str(m) for m in members if m]
        for fid, members in data.get("families", {}).items()
        if isinstance(members, list)
    }
    return device_to_family, families


# ---------------------------------------------------------------------------
# ES retrieval helpers
# ---------------------------------------------------------------------------


def _build_es_client():
    from elasticsearch import Elasticsearch
    from backend.config.settings import search_settings

    if search_settings.es_user and search_settings.es_password:
        return Elasticsearch(
            hosts=[search_settings.es_host],
            verify_certs=True,
            basic_auth=(search_settings.es_user, search_settings.es_password),
        )
    return Elasticsearch(hosts=[search_settings.es_host], verify_certs=True)


def _build_retrieval_runner(corpus_doc_ids: list[str], index_name: str | None = None):
    """Build a minimal retrieval runner for BM25 with optional extra_filter."""
    from backend.config.settings import search_settings
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine

    es_client = _build_es_client()
    if index_name is None:
        idx = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"
    else:
        idx = index_name

    engine = EsSearchEngine(
        es_client=es_client,
        index_name=idx,
        text_fields=["search_text^1.0", "chunk_summary^0.7", "chunk_keywords^0.8"],
    )
    corpus_filter = engine.build_filter(doc_ids=corpus_doc_ids)
    if corpus_filter is None:
        raise RuntimeError("Failed to build corpus filter")
    return engine, corpus_filter


def _merge_filters(
    base: dict[str, Any], extra: dict[str, Any] | None
) -> dict[str, Any]:
    if extra is None:
        return base
    return {"bool": {"filter": [base, extra]}}


def _run_bm25(
    engine: Any,
    base_filter: dict[str, Any],
    query: str,
    extra_filter: dict[str, Any] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    final = _merge_filters(base_filter, extra_filter)
    hits = engine.sparse_search(query_text=query, top_k=top_k, filters=final)
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, 1):
        meta = hit.metadata if isinstance(hit.metadata, dict) else {}
        rows.append(
            {
                "rank": rank,
                "doc_id": hit.doc_id,
                "score": float(hit.score),
                "device_name": str(meta.get("device_name") or ""),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_retrieval_metrics(
    doc_rows: list[dict[str, Any]],
    gold_doc_ids: list[str],
    allowed_devices: set[str],
    shared_doc_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    gold_set = set(gold_doc_ids)
    top_rows = doc_rows[:k]

    # MRR
    mrr = 0.0
    for idx, row in enumerate(doc_rows, 1):
        if row["doc_id"] in gold_set:
            mrr = 1.0 / float(idx)
            break

    # gold_hit@k
    top_ids = {r["doc_id"] for r in top_rows}
    gold_hit = int(bool(top_ids & gold_set))

    # contamination@k
    adj_oos = 0
    shared_count = 0
    for row in top_rows:
        doc_id = row["doc_id"]
        doc_device = str(row.get("device_name") or "").strip()
        in_scope = doc_device in allowed_devices if doc_device else False
        is_shared = doc_id in shared_doc_ids
        if is_shared:
            shared_count += 1
        if not in_scope and not is_shared:
            adj_oos += 1

    adj_den = k - shared_count
    adj_cont = float(adj_oos) / max(adj_den, 1) if adj_den > 0 else 0.0

    return {
        f"gold_hit@{k}": float(gold_hit),
        f"adj_cont@{k}": adj_cont,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading artifacts...")

    eval_rows: list[dict[str, Any]] = [
        row for row in read_jsonl(EVAL_PATH) if isinstance(row, dict)
    ]
    print(f"  eval rows: {len(eval_rows)}")

    doc_scope_rows = _load_doc_scope(DOC_SCOPE_PATH)
    shared_doc_ids: set[str] = set()
    if SHARED_DOC_IDS_PATH.exists():
        shared_doc_ids = set(_load_lines(SHARED_DOC_IDS_PATH))
    # also from doc_scope is_shared
    for row in doc_scope_rows:
        if row.get("is_shared") and row.get("es_doc_id"):
            shared_doc_ids.add(str(row["es_doc_id"]).strip())
    print(f"  shared_doc_ids: {len(shared_doc_ids)}")

    corpus_doc_ids = _load_lines(CORPUS_FILTER_PATH)
    print(f"  corpus_doc_ids: {len(corpus_doc_ids)}")

    device_to_family, families = _load_family_map(FAMILY_MAP_PATH)
    known_equip_ids = {
        str(row.get("canonical_equip_id") or "").strip().upper()
        for row in eval_rows
        if str(row.get("canonical_equip_id") or "").strip()
    }

    # Build device candidates list from doc_scope
    device_candidates: list[str] = sorted(
        {
            str(row.get("es_device_name") or "").strip()
            for row in doc_scope_rows
            if str(row.get("es_device_name") or "").strip()
        }
    )
    print(f"  device_candidates: {device_candidates}")

    # Build doc_id -> device_name map
    doc_device_map: dict[str, str] = {
        str(row["es_doc_id"]).strip(): str(row.get("es_device_name") or "").strip()
        for row in doc_scope_rows
        if row.get("es_doc_id")
    }

    # ---------------------------------------------------------------------------
    # Step 1: Parser accuracy
    # ---------------------------------------------------------------------------
    print("\n=== Step 1: Parser Accuracy ===")

    parser_results: list[dict[str, Any]] = []
    scope_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "exact_match": 0, "no_detection": 0, "wrong_detection": 0}
    )

    for row in eval_rows:
        q_id = str(row.get("q_id") or "")
        question = str(row.get("question") or "")
        gold_device = str(row.get("canonical_device_name") or "").strip()
        gold_equip = str(row.get("canonical_equip_id") or "").strip().upper()
        scope_obs = str(row.get("scope_observability") or "unknown")

        parsed = parse_devices_from_query(device_candidates, question)
        parsed_equip_ids = parse_equip_ids_from_query(known_equip_ids, question)
        parsed_device = parsed[0] if parsed else None
        parsed_equip = parsed_equip_ids[0] if parsed_equip_ids else None

        exact_match = bool(parsed_device and parsed_device == gold_device)
        equip_exact_match = bool(parsed_equip and parsed_equip == gold_equip)
        no_detection = parsed_device is None
        wrong_detection = bool(parsed_device and parsed_device != gold_device)

        parser_results.append(
            {
                "q_id": q_id,
                "question": question,
                "gold_device": gold_device,
                "gold_equip": gold_equip,
                "parsed_device": parsed_device,
                "parsed_equip": parsed_equip,
                "scope_observability": scope_obs,
                "exact_match": exact_match,
                "equip_exact_match": equip_exact_match,
                "no_detection": no_detection,
                "wrong_detection": wrong_detection,
            }
        )

        stats = scope_stats[scope_obs]
        stats["total"] += 1
        if exact_match:
            stats["exact_match"] += 1
        if no_detection:
            stats["no_detection"] += 1
        if wrong_detection:
            stats["wrong_detection"] += 1

    total = len(parser_results)
    overall_exact = sum(1 for r in parser_results if r["exact_match"])
    overall_no_det = sum(1 for r in parser_results if r["no_detection"])
    overall_wrong = sum(1 for r in parser_results if r["wrong_detection"])

    print(f"\nOverall parser accuracy ({total} queries):")
    print(f"  Exact match:     {overall_exact}/{total} = {overall_exact / total:.1%}")
    print(f"  No detection:    {overall_no_det}/{total} = {overall_no_det / total:.1%}")
    print(f"  Wrong detection: {overall_wrong}/{total} = {overall_wrong / total:.1%}")

    print("\nBy scope_observability:")
    for obs, stats in sorted(scope_stats.items()):
        n = stats["total"]
        em = stats["exact_match"]
        nd = stats["no_detection"]
        wd = stats["wrong_detection"]
        print(
            f"  {obs:25s}: total={n:3d}  exact={em:3d} ({em / n:.1%})  no_det={nd:3d} ({nd / n:.1%})  wrong={wd:3d} ({wd / n:.1%})"
        )

    # ---------------------------------------------------------------------------
    # Step 2: Build ES retrieval runner (requires ES to be running)
    # ---------------------------------------------------------------------------
    print("\n=== Step 2: Oracle B4 vs Real B4 Retrieval ===")

    engine: Any | None = None
    base_filter: dict[str, Any] | None = None
    fallback_count = 0
    parsed_count = 0
    scope_fallback_count = 0
    scope_parsed_device_count = 0
    scope_parsed_equip_count = 0

    try:
        engine, base_filter = _build_retrieval_runner(corpus_doc_ids)
        print("  ES connection OK")
        es_available = True
    except Exception as exc:
        print(f"  ES not available: {exc}")
        print("  Skipping retrieval comparison (offline mode)")
        es_available = False

    oracle_metrics_list: list[dict[str, Any]] = []
    real_metrics_list: list[dict[str, Any]] = []
    real_scope_metrics_list: list[dict[str, Any]] = []

    if es_available and engine is not None and base_filter is not None:
        # Only evaluate rows that have gold_doc_ids and allowed_devices
        eval_rows_with_gold = [
            r for r in eval_rows if r.get("gold_doc_ids") and r.get("allowed_devices")
        ]
        print(
            f"  Evaluating {len(eval_rows_with_gold)} queries with gold_doc_ids + allowed_devices"
        )

        # Build a q_id -> parser_result lookup
        parser_map = {r["q_id"]: r for r in parser_results}

        for i, row in enumerate(eval_rows_with_gold):
            q_id = str(row.get("q_id") or "")
            question = str(row.get("question") or "")
            gold_doc_ids = [str(d) for d in (row.get("gold_doc_ids") or []) if d]
            allowed_devices = set(
                str(d) for d in (row.get("allowed_devices") or []) if d
            )
            gold_device = str(row.get("canonical_device_name") or "").strip()
            scope_obs = str(row.get("scope_observability") or "unknown")

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(eval_rows_with_gold)}")

            # Oracle B4: use gold device as filter
            if gold_device:
                oracle_filter = engine.build_filter(device_names=[gold_device])
            else:
                oracle_filter = None

            t0 = time.monotonic()
            oracle_rows = _run_bm25(engine, base_filter, question, oracle_filter, TOP_K)
            oracle_latency = (time.monotonic() - t0) * 1000

            oracle_m = _compute_retrieval_metrics(
                oracle_rows, gold_doc_ids, allowed_devices, shared_doc_ids, TOP_K
            )
            oracle_metrics_list.append(
                {
                    "q_id": q_id,
                    "scope_observability": scope_obs,
                    "gold_device": gold_device,
                    "filter_device": gold_device,
                    "filter_type": "oracle",
                    "latency_ms": oracle_latency,
                    **oracle_m,
                }
            )

            # Real B4: use parser-detected device as filter
            pr = parser_map.get(q_id, {})
            parsed_device = pr.get("parsed_device")
            parsed_equip = pr.get("parsed_equip")

            if parsed_device:
                real_filter = engine.build_filter(device_names=[parsed_device])
                filter_type = "parsed"
            else:
                real_filter = None
                filter_type = "no_filter_fallback"

            t0 = time.monotonic()
            real_rows = _run_bm25(engine, base_filter, question, real_filter, TOP_K)
            real_latency = (time.monotonic() - t0) * 1000

            real_m = _compute_retrieval_metrics(
                real_rows, gold_doc_ids, allowed_devices, shared_doc_ids, TOP_K
            )
            real_metrics_list.append(
                {
                    "q_id": q_id,
                    "scope_observability": scope_obs,
                    "gold_device": gold_device,
                    "filter_device": parsed_device,
                    "filter_type": filter_type,
                    "latency_ms": real_latency,
                    **real_m,
                }
            )

            if parsed_equip:
                scope_filter = engine.build_filter(equip_ids=[str(parsed_equip)])
                scope_filter_type = "parsed_equip"
            elif parsed_device:
                scope_filter = engine.build_filter(device_names=[str(parsed_device)])
                scope_filter_type = "parsed_device"
            else:
                scope_filter = None
                scope_filter_type = "no_filter_fallback"

            t0 = time.monotonic()
            scope_rows = _run_bm25(engine, base_filter, question, scope_filter, TOP_K)
            scope_latency = (time.monotonic() - t0) * 1000
            scope_m = _compute_retrieval_metrics(
                scope_rows, gold_doc_ids, allowed_devices, shared_doc_ids, TOP_K
            )
            real_scope_metrics_list.append(
                {
                    "q_id": q_id,
                    "scope_observability": scope_obs,
                    "gold_device": gold_device,
                    "gold_equip": str(row.get("canonical_equip_id") or "")
                    .strip()
                    .upper(),
                    "filter_device": parsed_device,
                    "filter_equip": parsed_equip,
                    "filter_type": scope_filter_type,
                    "latency_ms": scope_latency,
                    **scope_m,
                }
            )

        # Aggregate comparison
        def _avg(rows: list[dict[str, Any]], key: str) -> float:
            vals = [r[key] for r in rows if key in r]
            return sum(vals) / len(vals) if vals else 0.0

        print("\n--- Oracle B4 vs Real B4 (BM25, all queries with gold) ---")
        print(f"{'Metric':<20} {'Oracle B4':>12} {'Real B4':>12} {'Delta':>10}")
        print("-" * 56)
        for metric in [f"gold_hit@{TOP_K}", f"adj_cont@{TOP_K}", "mrr"]:
            oracle_avg = _avg(oracle_metrics_list, metric)
            real_avg = _avg(real_metrics_list, metric)
            delta = real_avg - oracle_avg
            print(
                f"  {metric:<18} {oracle_avg:>12.4f} {real_avg:>12.4f} {delta:>+10.4f}"
            )

        # By scope_observability
        print("\n--- By scope_observability ---")
        obs_vals = sorted({r["scope_observability"] for r in oracle_metrics_list})
        for obs in obs_vals:
            o_rows = [r for r in oracle_metrics_list if r["scope_observability"] == obs]
            r_rows = [r for r in real_metrics_list if r["scope_observability"] == obs]
            n = len(o_rows)
            o_hit = _avg(o_rows, f"gold_hit@{TOP_K}")
            r_hit = _avg(r_rows, f"gold_hit@{TOP_K}")
            o_cont = _avg(o_rows, f"adj_cont@{TOP_K}")
            r_cont = _avg(r_rows, f"adj_cont@{TOP_K}")
            print(
                f"  {obs:25s} (n={n:3d})  gold_hit: oracle={o_hit:.3f}  real={r_hit:.3f}  delta={r_hit - o_hit:+.3f}  |  adj_cont: oracle={o_cont:.3f}  real={r_cont:.3f}  delta={r_cont - o_cont:+.3f}"
            )

        # fallback stats
        fallback_count = sum(
            1 for r in real_metrics_list if r["filter_type"] == "no_filter_fallback"
        )
        parsed_count = sum(1 for r in real_metrics_list if r["filter_type"] == "parsed")
        print(
            f"\n  Filter type breakdown: parsed={parsed_count}, no_filter_fallback={fallback_count}"
        )

        scope_fallback_count = sum(
            1
            for r in real_scope_metrics_list
            if r["filter_type"] == "no_filter_fallback"
        )
        scope_parsed_device_count = sum(
            1 for r in real_scope_metrics_list if r["filter_type"] == "parsed_device"
        )
        scope_parsed_equip_count = sum(
            1 for r in real_scope_metrics_list if r["filter_type"] == "parsed_equip"
        )

        print(
            "\n--- Oracle B4 vs Real B4 (equip-aware parser, BM25, all queries with gold) ---"
        )
        print(f"{'Metric':<20} {'Oracle B4':>12} {'Equip-aware':>12} {'Delta':>10}")
        print("-" * 56)
        for metric in [f"gold_hit@{TOP_K}", f"adj_cont@{TOP_K}", "mrr"]:
            oracle_avg = _avg(oracle_metrics_list, metric)
            scope_avg = _avg(real_scope_metrics_list, metric)
            delta = scope_avg - oracle_avg
            print(
                f"  {metric:<18} {oracle_avg:>12.4f} {scope_avg:>12.4f} {delta:>+10.4f}"
            )
        print(
            "\n  Filter type breakdown: "
            + f"parsed_device={scope_parsed_device_count}, "
            + f"parsed_equip={scope_parsed_equip_count}, "
            + f"no_filter_fallback={scope_fallback_count}"
        )

    # ---------------------------------------------------------------------------
    # Step 3: Save report
    # ---------------------------------------------------------------------------
    report: dict[str, Any] = {
        "meta": {
            "eval_path": str(EVAL_PATH),
            "doc_scope_path": str(DOC_SCOPE_PATH),
            "corpus_filter_path": str(CORPUS_FILTER_PATH),
            "top_k": TOP_K,
            "total_queries": total,
        },
        "parser_accuracy": {
            "overall": {
                "total": total,
                "exact_match": overall_exact,
                "exact_match_rate": round(overall_exact / total, 4) if total else 0.0,
                "no_detection": overall_no_det,
                "no_detection_rate": round(overall_no_det / total, 4) if total else 0.0,
                "wrong_detection": overall_wrong,
                "wrong_detection_rate": round(overall_wrong / total, 4)
                if total
                else 0.0,
            },
            "equip_overall": {
                "total_with_equip": sum(1 for r in parser_results if r["gold_equip"]),
                "exact_match": sum(1 for r in parser_results if r["equip_exact_match"]),
            },
            "by_scope_observability": {
                obs: {
                    "total": stats["total"],
                    "exact_match": stats["exact_match"],
                    "exact_match_rate": round(stats["exact_match"] / stats["total"], 4)
                    if stats["total"]
                    else 0.0,
                    "no_detection": stats["no_detection"],
                    "no_detection_rate": round(
                        stats["no_detection"] / stats["total"], 4
                    )
                    if stats["total"]
                    else 0.0,
                    "wrong_detection": stats["wrong_detection"],
                    "wrong_detection_rate": round(
                        stats["wrong_detection"] / stats["total"], 4
                    )
                    if stats["total"]
                    else 0.0,
                }
                for obs, stats in sorted(scope_stats.items())
            },
        },
        "parser_per_query": parser_results,
        "retrieval_oracle_b4": oracle_metrics_list if es_available else [],
        "retrieval_real_b4": real_metrics_list if es_available else [],
        "retrieval_real_b4_scope_aware": real_scope_metrics_list
        if es_available
        else [],
    }

    if es_available and oracle_metrics_list:

        def _avg_safe(rows: list[dict[str, Any]], key: str) -> float:
            vals = [r[key] for r in rows if key in r]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        obs_comparison: dict[str, Any] = {}
        obs_vals_all = sorted({r["scope_observability"] for r in oracle_metrics_list})
        for obs in obs_vals_all:
            o_rows = [r for r in oracle_metrics_list if r["scope_observability"] == obs]
            r_rows = [r for r in real_metrics_list if r["scope_observability"] == obs]
            obs_comparison[obs] = {
                "n": len(o_rows),
                "oracle_b4": {
                    f"gold_hit@{TOP_K}": _avg_safe(o_rows, f"gold_hit@{TOP_K}"),
                    f"adj_cont@{TOP_K}": _avg_safe(o_rows, f"adj_cont@{TOP_K}"),
                    "mrr": _avg_safe(o_rows, "mrr"),
                },
                "real_b4": {
                    f"gold_hit@{TOP_K}": _avg_safe(r_rows, f"gold_hit@{TOP_K}"),
                    f"adj_cont@{TOP_K}": _avg_safe(r_rows, f"adj_cont@{TOP_K}"),
                    "mrr": _avg_safe(r_rows, "mrr"),
                },
            }

        report["retrieval_comparison"] = {
            "overall": {
                "n": len(oracle_metrics_list),
                "oracle_b4": {
                    f"gold_hit@{TOP_K}": _avg_safe(
                        oracle_metrics_list, f"gold_hit@{TOP_K}"
                    ),
                    f"adj_cont@{TOP_K}": _avg_safe(
                        oracle_metrics_list, f"adj_cont@{TOP_K}"
                    ),
                    "mrr": _avg_safe(oracle_metrics_list, "mrr"),
                },
                "real_b4": {
                    f"gold_hit@{TOP_K}": _avg_safe(
                        real_metrics_list, f"gold_hit@{TOP_K}"
                    ),
                    f"adj_cont@{TOP_K}": _avg_safe(
                        real_metrics_list, f"adj_cont@{TOP_K}"
                    ),
                    "mrr": _avg_safe(real_metrics_list, "mrr"),
                },
                "fallback_count": fallback_count,
                "parsed_count": parsed_count,
            },
            "by_scope_observability": obs_comparison,
        }

        report["retrieval_comparison_scope_aware"] = {
            "overall": {
                "n": len(real_scope_metrics_list),
                "oracle_b4": {
                    f"gold_hit@{TOP_K}": _avg_safe(
                        oracle_metrics_list, f"gold_hit@{TOP_K}"
                    ),
                    f"adj_cont@{TOP_K}": _avg_safe(
                        oracle_metrics_list, f"adj_cont@{TOP_K}"
                    ),
                    "mrr": _avg_safe(oracle_metrics_list, "mrr"),
                },
                "real_b4_scope_aware": {
                    f"gold_hit@{TOP_K}": _avg_safe(
                        real_scope_metrics_list, f"gold_hit@{TOP_K}"
                    ),
                    f"adj_cont@{TOP_K}": _avg_safe(
                        real_scope_metrics_list, f"adj_cont@{TOP_K}"
                    ),
                    "mrr": _avg_safe(real_scope_metrics_list, "mrr"),
                },
                "parsed_device_count": scope_parsed_device_count,
                "parsed_equip_count": scope_parsed_equip_count,
                "fallback_count": scope_fallback_count,
            }
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nReport saved to: {OUT_PATH}")

    _write_diff_csv(parser_results, oracle_metrics_list, real_metrics_list)
    _write_evidence_markdown(report)


def _write_diff_csv(
    parser_results: list[dict[str, Any]],
    oracle_metrics_list: list[dict[str, Any]],
    real_metrics_list: list[dict[str, Any]],
) -> None:
    def _as_row_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    parser_map = {str(row["q_id"]): row for row in parser_results}
    oracle_map = {str(row["q_id"]): row for row in oracle_metrics_list}
    real_map = {str(row["q_id"]): row for row in real_metrics_list}

    fieldnames = [
        "q_id",
        "scope_observability",
        "gold_device",
        "parsed_device",
        "parser_device_correct",
        "parser_no_detection",
        "oracle_hit",
        "parser_hit",
        "oracle_adj_cont",
        "parser_adj_cont",
        "oracle_mrr",
        "parser_mrr",
        "oracle_filter_device",
        "parser_filter_device",
        "parser_filter_type",
        "hit_delta",
        "adj_cont_delta",
        "mrr_delta",
    ]

    DIFF_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DIFF_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        q_ids = sorted(set(parser_map) | set(oracle_map) | set(real_map))
        for q_id in q_ids:
            parser_row = _as_row_dict(parser_map.get(q_id))
            oracle_row = _as_row_dict(oracle_map.get(q_id))
            real_row = _as_row_dict(real_map.get(q_id))
            oracle_hit = float(oracle_row.get(f"gold_hit@{TOP_K}", 0.0))
            parser_hit = float(real_row.get(f"gold_hit@{TOP_K}", 0.0))
            oracle_adj_cont = float(oracle_row.get(f"adj_cont@{TOP_K}", 0.0))
            parser_adj_cont = float(real_row.get(f"adj_cont@{TOP_K}", 0.0))
            oracle_mrr = float(oracle_row.get("mrr", 0.0))
            parser_mrr = float(real_row.get("mrr", 0.0))

            writer.writerow(
                {
                    "q_id": q_id,
                    "scope_observability": parser_row.get(
                        "scope_observability", oracle_row.get("scope_observability", "")
                    ),
                    "gold_device": parser_row.get(
                        "gold_device", oracle_row.get("gold_device", "")
                    ),
                    "parsed_device": parser_row.get("parsed_device") or "",
                    "parser_device_correct": parser_row.get("exact_match", False),
                    "parser_no_detection": parser_row.get("no_detection", False),
                    "oracle_hit": oracle_hit,
                    "parser_hit": parser_hit,
                    "oracle_adj_cont": oracle_adj_cont,
                    "parser_adj_cont": parser_adj_cont,
                    "oracle_mrr": oracle_mrr,
                    "parser_mrr": parser_mrr,
                    "oracle_filter_device": oracle_row.get("filter_device", ""),
                    "parser_filter_device": real_row.get("filter_device", ""),
                    "parser_filter_type": real_row.get("filter_type", ""),
                    "hit_delta": round(parser_hit - oracle_hit, 4),
                    "adj_cont_delta": round(parser_adj_cont - oracle_adj_cont, 4),
                    "mrr_delta": round(parser_mrr - oracle_mrr, 4),
                }
            )

    print(f"Per-query diff CSV saved to: {DIFF_CSV_PATH}")


def _write_evidence_markdown(report: dict[str, Any]) -> None:
    parser_accuracy = report.get("parser_accuracy")
    if not isinstance(parser_accuracy, dict):
        raise ValueError("parser_accuracy section missing from report")
    parser_overall = parser_accuracy.get("overall")
    if not isinstance(parser_overall, dict):
        raise ValueError("parser_accuracy.overall section missing from report")

    retrieval_raw = report.get("retrieval_comparison")
    retrieval = retrieval_raw if isinstance(retrieval_raw, dict) else {}
    retrieval_overall_raw = retrieval.get("overall")
    retrieval_overall = (
        retrieval_overall_raw if isinstance(retrieval_overall_raw, dict) else {}
    )
    by_scope_raw = retrieval.get("by_scope_observability")
    by_scope = by_scope_raw if isinstance(by_scope_raw, dict) else {}
    retrieval_scope_raw = report.get("retrieval_comparison_scope_aware")
    retrieval_scope = (
        retrieval_scope_raw if isinstance(retrieval_scope_raw, dict) else {}
    )
    retrieval_scope_overall_raw = retrieval_scope.get("overall")
    retrieval_scope_overall = (
        retrieval_scope_overall_raw
        if isinstance(retrieval_scope_overall_raw, dict)
        else {}
    )

    def _pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines = [
        "# Oracle vs Parser Gap (2026-03-14)",
        "",
        "Date: 2026-03-14",
        "Status: generated from `scripts/paper_a/measure_parser_accuracy.py`",
        "",
        "## Inputs",
        "",
        f"- Eval set: `{EVAL_PATH.relative_to(ROOT)}`",
        f"- Doc scope: `{DOC_SCOPE_PATH.relative_to(ROOT)}`",
        f"- Shared doc ids: `{SHARED_DOC_IDS_PATH.relative_to(ROOT)}`",
        f"- Corpus filter: `{CORPUS_FILTER_PATH.relative_to(ROOT)}`",
        "",
        "## Command",
        "",
        "```bash",
        "cd /home/hskim/work/llm-agent-v2",
        "uv run python scripts/paper_a/measure_parser_accuracy.py",
        "```",
        "",
        "## Parser Accuracy",
        "",
        f"- Total queries: {parser_overall['total']}",
        f"- Exact match: {parser_overall['exact_match']}/{parser_overall['total']} ({_pct(parser_overall['exact_match_rate'])})",
        f"- No detection: {parser_overall['no_detection']}/{parser_overall['total']} ({_pct(parser_overall['no_detection_rate'])})",
        f"- Wrong detection: {parser_overall['wrong_detection']}/{parser_overall['total']} ({_pct(parser_overall['wrong_detection_rate'])})",
        "",
        "## Oracle vs Real Parser Retrieval (BM25, top-10)",
        "",
    ]

    if retrieval_overall:
        oracle = retrieval_overall.get("oracle_b4", {})
        real = retrieval_overall.get("real_b4", {})
        lines.extend(
            [
                f"- Oracle gold_hit@{TOP_K}: {_pct(float(oracle.get(f'gold_hit@{TOP_K}', 0.0)))}",
                f"- Real parser gold_hit@{TOP_K}: {_pct(float(real.get(f'gold_hit@{TOP_K}', 0.0)))}",
                f"- Delta: {(float(real.get(f'gold_hit@{TOP_K}', 0.0)) - float(oracle.get(f'gold_hit@{TOP_K}', 0.0))) * 100:+.1f}%p",
                f"- Oracle adj_cont@{TOP_K}: {_pct(float(oracle.get(f'adj_cont@{TOP_K}', 0.0)))}",
                f"- Real parser adj_cont@{TOP_K}: {_pct(float(real.get(f'adj_cont@{TOP_K}', 0.0)))}",
                f"- Parsed filters: {retrieval_overall.get('parsed_count', 0)}",
                f"- No-filter fallback: {retrieval_overall.get('fallback_count', 0)}",
                "",
                "## By scope_observability",
                "",
            ]
        )
        for obs in sorted(by_scope):
            obs_item_raw = by_scope[obs]
            if not isinstance(obs_item_raw, dict):
                continue
            obs_item = obs_item_raw
            oracle_obs_raw = obs_item.get("oracle_b4")
            real_obs_raw = obs_item.get("real_b4")
            oracle_obs = oracle_obs_raw if isinstance(oracle_obs_raw, dict) else {}
            real_obs = real_obs_raw if isinstance(real_obs_raw, dict) else {}
            lines.append(
                "- "
                + f"{obs} (n={obs_item.get('n', 0)}): gold_hit {_pct(float(oracle_obs.get(f'gold_hit@{TOP_K}', 0.0)))} -> {_pct(float(real_obs.get(f'gold_hit@{TOP_K}', 0.0)))}, "
                + f"adj_cont {_pct(float(oracle_obs.get(f'adj_cont@{TOP_K}', 0.0)))} -> {_pct(float(real_obs.get(f'adj_cont@{TOP_K}', 0.0)))}"
            )
        if retrieval_scope_overall:
            scope_real = retrieval_scope_overall.get("real_b4_scope_aware", {})
            if isinstance(scope_real, dict):
                lines.extend(
                    [
                        "",
                        "## Equip-aware realistic mode",
                        "",
                        f"- Scope-aware parser gold_hit@{TOP_K}: {_pct(float(scope_real.get(f'gold_hit@{TOP_K}', 0.0)))}",
                        f"- Scope-aware parser adj_cont@{TOP_K}: {_pct(float(scope_real.get(f'adj_cont@{TOP_K}', 0.0)))}",
                        f"- Parsed device filters: {retrieval_scope_overall.get('parsed_device_count', 0)}",
                        f"- Parsed equip filters: {retrieval_scope_overall.get('parsed_equip_count', 0)}",
                        f"- No-filter fallback: {retrieval_scope_overall.get('fallback_count', 0)}",
                    ]
                )
    else:
        lines.append("- Retrieval comparison skipped because ES was unavailable.")

    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            f"- JSON report: `{OUT_PATH.relative_to(ROOT)}`",
            f"- Per-query diff CSV: `{DIFF_CSV_PATH.relative_to(ROOT)}`",
            "",
            "## Interpretation Notes",
            "",
            "- Oracle numbers are upper bounds and must not be reported as realistic production performance.",
            "- Device-only parsing fails mainly on explicit_equip rows; the equip-aware comparison is the more realistic upper baseline for those queries.",
            "- Use the per-query CSV to inspect cases where parser failure preserves hit but sharply increases contamination.",
        ]
    )

    EVIDENCE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evidence markdown saved to: {EVIDENCE_MD_PATH}")


if __name__ == "__main__":
    main()
