"""Build Gold Doc Expansion Candidate Pack for Paper A.

Reads query_gold_master JSONL, finds queries with empty gold_doc_ids,
and outputs a JSON report + Markdown checklist for PE review.

No ES dependency. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import cast

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
        "--top-n",
        type=int,
        default=30,
        help="Number of candidates per scope group to include (default: 30)",
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "total_queries": total_queries,
        "empty_gold_count": empty_gold_count,
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
                f"- [ ] q_id: `{q_id}` | question: \"{question}\" | devices: [{devices_str}]"
            )
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def run() -> int:
    args = parse_args()
    eval_set_path = Path(cast(str, args.eval_set))
    out_json_path = Path(cast(str, args.out_json))
    out_md_path = Path(cast(str, args.out_md))
    top_n = int(args.top_n)
    seed: int | None = args.seed

    if not eval_set_path.exists():
        print(f"ERROR: eval-set file not found: {eval_set_path}", file=sys.stderr)
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

    try:
        _write_json(out_json_path, total_queries, empty_gold_count, candidates, generated_at)
        _write_md(out_md_path, total_queries, empty_gold_count, candidates, generated_at)
    except Exception as exc:
        print(f"ERROR: failed to write outputs: {exc}", file=sys.stderr)
        return 1

    print(f"total_queries   : {total_queries}")
    print(f"empty_gold_count: {empty_gold_count} ({empty_gold_count/total_queries*100:.1f}%)")
    print(f"candidates       : {len(candidates)}")
    print(f"out_json         : {out_json_path}")
    print(f"out_md           : {out_md_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
