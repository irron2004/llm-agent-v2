#!/usr/bin/env python3
"""통합 Chat-Flow 평가 스크립트.

모든 task_mode (sop, ts, issue, general)를 하나의 스크립트로 평가한다.
두 가지 모드 지원:
  --mode direct : LangGraphRAGAgent 직접 호출 (ES 필요, API 서버 불필요)
  --mode http   : HTTP API 호출 (API 서버 필요)

Usage:
  # 1) eval 데이터셋 먼저 생성
  python scripts/evaluation/generate_chatflow_eval_dataset.py

  # 2-a) Direct mode (ES만 필요)
  python scripts/evaluation/evaluate_chatflow_unified.py \
    --input data/eval_chatflow_unified.jsonl \
    --out-dir data/eval_results/chatflow_unified_$(date +%Y%m%d) \
    --mode direct --limit 20

  # 2-b) HTTP mode (API 서버 필요)
  python scripts/evaluation/evaluate_chatflow_unified.py \
    --input data/eval_chatflow_unified.jsonl \
    --out-dir data/eval_results/chatflow_unified_$(date +%Y%m%d) \
    --mode http --api-base-url http://localhost:8011 --limit 20
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalQuery:
    q_id: str
    question: str
    split: str
    intent_primary: str
    scope_observability: str
    canonical_device_name: str
    canonical_equip_id: str
    preferred_doc_types: list[str]
    gold_doc_ids: list[str]
    gold_doc_types: list[str]
    expected_task_mode: str
    expected_route: str


@dataclass
class EvalResult:
    q_id: str
    question: str
    expected_task_mode: str
    expected_route: str
    actual_route: str | None
    actual_task_mode: str | None
    route_correct: bool
    task_mode_correct: bool
    gold_doc_ids: list[str]
    hit_doc_at_5: bool
    hit_doc_at_10: bool
    hit_count_at_10: int
    top10_doc_ids: list[str]
    elapsed_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip().replace(" ", "_").replace("-", "_")


def _doc_matches(gold_doc: str, candidate: str) -> bool:
    g = _normalize(gold_doc)
    c = _normalize(candidate)
    return g in c or c in g


def _compute_hits(
    gold_doc_ids: list[str], retrieved_doc_ids: list[str], k: int
) -> tuple[bool, int]:
    """Check if any gold doc appears in top-k retrieved docs."""
    hit_count = 0
    for rdoc in retrieved_doc_ids[:k]:
        for gdoc in gold_doc_ids:
            if _doc_matches(gdoc, rdoc):
                hit_count += 1
                break
    return hit_count > 0, hit_count


def _load_eval_queries(path: Path, limit: int | None, splits: list[str] | None) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if splits and row.get("split") not in splits:
                continue
            queries.append(EvalQuery(
                q_id=row["q_id"],
                question=row["question"],
                split=row.get("split", ""),
                intent_primary=row.get("intent_primary", ""),
                scope_observability=row.get("scope_observability", ""),
                canonical_device_name=row.get("canonical_device_name", ""),
                canonical_equip_id=row.get("canonical_equip_id", ""),
                preferred_doc_types=row.get("preferred_doc_types", []),
                gold_doc_ids=row.get("gold_doc_ids", []),
                gold_doc_types=row.get("gold_doc_types", []),
                expected_task_mode=row.get("expected_task_mode", ""),
                expected_route=row.get("expected_route", ""),
            ))
            if limit and len(queries) >= limit:
                break
    return queries


# ---------------------------------------------------------------------------
# Direct mode: call LangGraphRAGAgent in-process
# ---------------------------------------------------------------------------

def _build_state_overrides(eq: EvalQuery) -> dict[str, Any]:
    """Build state_overrides based on expected task_mode.

    This simulates what the UI would send after device_selection/auto_parse.
    """
    overrides: dict[str, Any] = {
        "mq_mode": "fallback",
    }

    # Set device filter if explicit
    if eq.canonical_device_name:
        overrides["selected_devices"] = [eq.canonical_device_name]
    if eq.canonical_equip_id:
        overrides["selected_equip_ids"] = [eq.canonical_equip_id]

    # Set doc_type filter based on expected task_mode
    if eq.expected_task_mode == "sop":
        overrides["task_mode"] = "sop"
        overrides["selected_doc_types"] = ["sop", "set_up_manual"]
        overrides["selected_doc_types_strict"] = True
    elif eq.expected_task_mode == "ts":
        overrides["task_mode"] = "ts"
        overrides["selected_doc_types"] = ["trouble_shooting_guide"]
        overrides["selected_doc_types_strict"] = True
    elif eq.expected_task_mode == "issue":
        overrides["task_mode"] = "issue"
        # issue는 모든 doc_type 검색
    else:
        # general: no filter
        pass

    return overrides


def _run_direct(queries: list[EvalQuery], args: argparse.Namespace) -> list[EvalResult]:
    from backend.api.dependencies import (
        get_default_llm,
        get_prompt_spec_cached,
        get_search_service,
    )
    from backend.api.main import _configure_search_service
    from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec
    from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent
    from backend.services.search_service import SearchService

    _configure_search_service()
    llm = get_default_llm()
    search_service = cast(SearchService, cast(object, get_search_service()))
    prompt_spec = cast(PromptSpec, get_prompt_spec_cached())

    agent = LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=20,
        retrieval_top_k=50,
        mode="verified",
        ask_user_after_retrieve=True,  # stop before answer to get retrieval results
        auto_parse_enabled=False,
        use_canonical_retrieval=False,
    )

    results: list[EvalResult] = []
    total = len(queries)

    for idx, eq in enumerate(queries, 1):
        overrides = _build_state_overrides(eq)
        thread_id = f"eval-{eq.q_id}-{hashlib.sha1(eq.question.encode()).hexdigest()[:8]}"

        try:
            t0 = time.perf_counter()
            result = agent.run(
                eq.question,
                thread_id=thread_id,
                state_overrides=overrides,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            # Extract retrieved docs
            docs = result.get("docs") or result.get("retrieved_docs") or []
            top10_ids: list[str] = []
            for doc in docs[:10]:
                if isinstance(doc, dict):
                    top10_ids.append(str(doc.get("doc_id", "")))
                else:
                    doc_id = getattr(doc, "doc_id", None)
                    if doc_id is None:
                        md = getattr(doc, "metadata", {}) or {}
                        doc_id = md.get("doc_id", "")
                    top10_ids.append(str(doc_id))

            actual_route = result.get("route")
            actual_task_mode = result.get("task_mode")

            hit5, count5 = _compute_hits(eq.gold_doc_ids, top10_ids, 5)
            hit10, count10 = _compute_hits(eq.gold_doc_ids, top10_ids, 10)

            er = EvalResult(
                q_id=eq.q_id,
                question=eq.question,
                expected_task_mode=eq.expected_task_mode,
                expected_route=eq.expected_route,
                actual_route=actual_route,
                actual_task_mode=actual_task_mode,
                route_correct=(actual_route == eq.expected_route) if actual_route else False,
                task_mode_correct=(actual_task_mode == eq.expected_task_mode) if actual_task_mode else False,
                gold_doc_ids=eq.gold_doc_ids,
                hit_doc_at_5=hit5,
                hit_doc_at_10=hit10,
                hit_count_at_10=count10,
                top10_doc_ids=top10_ids,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            er = EvalResult(
                q_id=eq.q_id,
                question=eq.question,
                expected_task_mode=eq.expected_task_mode,
                expected_route=eq.expected_route,
                actual_route=None,
                actual_task_mode=None,
                route_correct=False,
                task_mode_correct=False,
                gold_doc_ids=eq.gold_doc_ids,
                hit_doc_at_5=False,
                hit_doc_at_10=False,
                hit_count_at_10=0,
                top10_doc_ids=[],
                elapsed_ms=0,
                error=str(exc),
            )

        results.append(er)
        status = "OK" if er.hit_doc_at_10 else "MISS"
        print(
            f"[{idx}/{total}] {status} {eq.q_id} mode={eq.expected_task_mode} "
            f"doc@5={'Y' if er.hit_doc_at_5 else 'N'} doc@10={'Y' if er.hit_doc_at_10 else 'N'} "
            f"{er.elapsed_ms/1000:.1f}s",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# HTTP mode: call API server
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    elapsed = (time.perf_counter() - t0) * 1000.0
    return json.loads(raw.decode("utf-8")), elapsed


def _run_http(queries: list[EvalQuery], args: argparse.Namespace) -> list[EvalResult]:
    base_url = args.api_base_url.rstrip("/")
    run_url = f"{base_url}/api/agent/run"

    results: list[EvalResult] = []
    total = len(queries)

    for idx, eq in enumerate(queries, 1):
        thread_id = f"eval-{eq.q_id}-{hashlib.sha1(eq.question.encode()).hexdigest()[:8]}"

        payload: dict[str, Any] = {
            "message": eq.question,
            "ask_user_after_retrieve": True,
            "auto_parse": False,
            "guided_confirm": False,
            "use_canonical_retrieval": False,
            "mq_mode": "fallback",
            "max_attempts": 0,
            "mode": "verified",
            "thread_id": thread_id,
        }

        try:
            response, elapsed = _post_json(run_url, payload, timeout=args.timeout)

            # Handle device_selection interrupt
            interrupt = response.get("interrupt_payload") or {}
            if response.get("interrupted") and interrupt.get("type") == "device_selection":
                # Auto-select the correct device
                device = eq.canonical_device_name
                if not device:
                    devices = interrupt.get("devices") or []
                    if devices:
                        device = devices[0].get("name", "")

                if device:
                    resume_payload: dict[str, Any] = {
                        **payload,
                        "thread_id": thread_id,
                        "resume_decision": {
                            "type": "device_selection",
                            "selected_devices": [device],
                            "selected_doc_types": eq.preferred_doc_types or [],
                        },
                    }
                    response2, elapsed2 = _post_json(run_url, resume_payload, timeout=args.timeout)
                    response = response2
                    elapsed += elapsed2

            # Extract results
            retrieved = response.get("retrieved_docs") or []
            top10_ids: list[str] = []
            for doc in retrieved[:10]:
                if isinstance(doc, dict):
                    doc_id = doc.get("doc_id") or doc.get("id") or ""
                    if not doc_id:
                        md = doc.get("metadata") or {}
                        doc_id = md.get("doc_id", "")
                    top10_ids.append(str(doc_id))

            metadata = response.get("metadata") or {}
            actual_route = metadata.get("route") or response.get("route")
            actual_task_mode = metadata.get("task_mode") or response.get("task_mode")

            hit5, _ = _compute_hits(eq.gold_doc_ids, top10_ids, 5)
            hit10, count10 = _compute_hits(eq.gold_doc_ids, top10_ids, 10)

            er = EvalResult(
                q_id=eq.q_id,
                question=eq.question,
                expected_task_mode=eq.expected_task_mode,
                expected_route=eq.expected_route,
                actual_route=actual_route,
                actual_task_mode=actual_task_mode,
                route_correct=(actual_route == eq.expected_route) if actual_route else False,
                task_mode_correct=(actual_task_mode == eq.expected_task_mode) if actual_task_mode else False,
                gold_doc_ids=eq.gold_doc_ids,
                hit_doc_at_5=hit5,
                hit_doc_at_10=hit10,
                hit_count_at_10=count10,
                top10_doc_ids=top10_ids,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            er = EvalResult(
                q_id=eq.q_id,
                question=eq.question,
                expected_task_mode=eq.expected_task_mode,
                expected_route=eq.expected_route,
                actual_route=None,
                actual_task_mode=None,
                route_correct=False,
                task_mode_correct=False,
                gold_doc_ids=eq.gold_doc_ids,
                hit_doc_at_5=False,
                hit_doc_at_10=False,
                hit_count_at_10=0,
                top10_doc_ids=[],
                elapsed_ms=0,
                error=str(exc),
            )

        results.append(er)
        status = "OK" if er.hit_doc_at_10 else "MISS"
        print(
            f"[{idx}/{total}] {status} {eq.q_id} mode={eq.expected_task_mode} "
            f"doc@5={'Y' if er.hit_doc_at_5 else 'N'} doc@10={'Y' if er.hit_doc_at_10 else 'N'} "
            f"{er.elapsed_ms/1000:.1f}s",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    queries: list[EvalQuery],
    results: list[EvalResult],
    out_dir: Path,
) -> None:
    # Overall metrics
    total = len(results)
    errors = sum(1 for r in results if r.error)
    valid = [r for r in results if not r.error]

    doc_hit5 = sum(1 for r in valid if r.hit_doc_at_5)
    doc_hit10 = sum(1 for r in valid if r.hit_doc_at_10)
    route_correct = sum(1 for r in valid if r.route_correct)
    avg_elapsed = sum(r.elapsed_ms for r in valid) / max(len(valid), 1)

    # Per task_mode breakdown
    by_mode: dict[str, list[EvalResult]] = {}
    for r in valid:
        by_mode.setdefault(r.expected_task_mode, []).append(r)

    # Per scope_observability
    query_map = {q.q_id: q for q in queries}
    by_scope: dict[str, list[EvalResult]] = {}
    for r in valid:
        scope = query_map.get(r.q_id, EvalQuery("","","","","","","",[], [], [], "", "")).scope_observability
        by_scope.setdefault(scope, []).append(r)

    # Write raw results
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fp:
        for r in results:
            fp.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Write summary JSON
    summary: dict[str, Any] = {
        "total": total,
        "errors": errors,
        "valid": len(valid),
        "doc_hit_at_5": doc_hit5,
        "doc_hit_at_5_pct": round(doc_hit5 / max(len(valid), 1) * 100, 2),
        "doc_hit_at_10": doc_hit10,
        "doc_hit_at_10_pct": round(doc_hit10 / max(len(valid), 1) * 100, 2),
        "route_accuracy": round(route_correct / max(len(valid), 1) * 100, 2),
        "avg_elapsed_ms": round(avg_elapsed),
        "by_task_mode": {},
        "by_scope": {},
    }

    for mode, items in sorted(by_mode.items()):
        n = len(items)
        summary["by_task_mode"][mode] = {
            "count": n,
            "doc_hit_at_5": sum(1 for r in items if r.hit_doc_at_5),
            "doc_hit_at_5_pct": round(sum(1 for r in items if r.hit_doc_at_5) / max(n, 1) * 100, 2),
            "doc_hit_at_10": sum(1 for r in items if r.hit_doc_at_10),
            "doc_hit_at_10_pct": round(sum(1 for r in items if r.hit_doc_at_10) / max(n, 1) * 100, 2),
            "route_correct": sum(1 for r in items if r.route_correct),
            "avg_elapsed_ms": round(sum(r.elapsed_ms for r in items) / max(n, 1)),
        }

    for scope, items in sorted(by_scope.items()):
        n = len(items)
        summary["by_scope"][scope] = {
            "count": n,
            "doc_hit_at_10": sum(1 for r in items if r.hit_doc_at_10),
            "doc_hit_at_10_pct": round(sum(1 for r in items if r.hit_doc_at_10) / max(n, 1) * 100, 2),
        }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # Write markdown report
    lines: list[str] = [
        "# Chat-Flow Unified Eval Report",
        "",
        f"- **Total queries**: {total} (errors: {errors})",
        f"- **doc_hit@5**: {doc_hit5}/{len(valid)} ({summary['doc_hit_at_5_pct']}%)",
        f"- **doc_hit@10**: {doc_hit10}/{len(valid)} ({summary['doc_hit_at_10_pct']}%)",
        f"- **route accuracy**: {route_correct}/{len(valid)} ({summary['route_accuracy']}%)",
        f"- **avg latency**: {avg_elapsed:.0f}ms",
        "",
        "## By Task Mode",
        "",
        "| task_mode | count | doc@5 | doc@10 | route_acc | avg_ms |",
        "|-----------|-------|-------|--------|-----------|--------|",
    ]
    for mode, stats in sorted(summary["by_task_mode"].items()):
        route_acc = round(stats["route_correct"] / max(stats["count"], 1) * 100, 1)
        lines.append(
            f"| {mode} | {stats['count']} | "
            f"{stats['doc_hit_at_5_pct']}% | {stats['doc_hit_at_10_pct']}% | "
            f"{route_acc}% | {stats['avg_elapsed_ms']} |"
        )

    lines.extend([
        "",
        "## By Scope Observability",
        "",
        "| scope | count | doc@10 |",
        "|-------|-------|--------|",
    ])
    for scope, stats in sorted(summary["by_scope"].items()):
        lines.append(f"| {scope} | {stats['count']} | {stats['doc_hit_at_10_pct']}% |")

    # Miss analysis
    misses = [r for r in valid if not r.hit_doc_at_10]
    if misses:
        lines.extend([
            "",
            "## Miss Analysis (doc@10 = N)",
            "",
            f"Total misses: {len(misses)}",
            "",
        ])
        miss_by_mode = Counter(r.expected_task_mode for r in misses)
        lines.append("By task_mode: " + ", ".join(f"{k}={v}" for k, v in miss_by_mode.most_common()))
        lines.append("")
        lines.append("| q_id | task_mode | question (50 chars) | top1_doc |")
        lines.append("|------|----------|---------------------|----------|")
        for r in misses[:30]:
            top1 = r.top10_doc_ids[0] if r.top10_doc_ids else "-"
            lines.append(f"| {r.q_id} | {r.expected_task_mode} | {r.question[:50]} | {top1} |")

    with (out_dir / "report.md").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"\nResults written to {out_dir}/")
    print(f"  doc_hit@5:  {doc_hit5}/{len(valid)} ({summary['doc_hit_at_5_pct']}%)")
    print(f"  doc_hit@10: {doc_hit10}/{len(valid)} ({summary['doc_hit_at_10_pct']}%)")
    print(f"  route_acc:  {route_correct}/{len(valid)} ({summary['route_accuracy']}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified chat-flow evaluation")
    parser.add_argument("--input", required=True, help="Eval dataset JSONL")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=("direct", "http"),
        default="direct",
        help="Execution mode",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max queries")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Filter by split (dev, test, dev_implicit)",
    )
    parser.add_argument("--api-base-url", default="http://localhost:8011")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_eval_queries(input_path, args.limit, args.splits)
    if not queries:
        print("No queries loaded.", file=sys.stderr)
        return 1

    print(f"Loaded {len(queries)} queries")
    mode_dist = Counter(q.expected_task_mode for q in queries)
    print(f"  task_mode distribution: {dict(mode_dist)}")

    if args.mode == "direct":
        results = _run_direct(queries, args)
    else:
        results = _run_http(queries, args)

    _generate_report(queries, results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
