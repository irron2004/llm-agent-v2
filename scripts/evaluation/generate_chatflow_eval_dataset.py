#!/usr/bin/env python3
"""gold_master에서 chat-flow 통합 eval 데이터셋 생성.

입력: data/paper_a/eval/query_gold_master_v0_7_with_implicit.jsonl
출력: data/eval_chatflow_unified.jsonl

각 질문에 대해 expected_route, expected_task_mode를 자동 추론한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _infer_gold_doc_types(gold_doc_ids: list[str]) -> set[str]:
    """gold_doc_id 패턴으로 문서 유형 추론."""
    types: set[str] = set()
    for doc_id in gold_doc_ids:
        dl = doc_id.lower()
        if "global_sop" in dl or dl.startswith("sop"):
            types.add("sop")
        elif "set_up_manual" in dl or "setup_manual" in dl:
            types.add("manual")
        elif "trouble_shooting" in dl or "_ts_" in dl or "_tsg_" in dl:
            types.add("ts")
        elif dl.startswith("gcb") or dl.startswith("gcb_"):
            types.add("gcb")
        else:
            # numeric doc_ids → myservice
            stripped = doc_id.replace("-", "").replace("_", "")
            if stripped.isdigit():
                types.add("myservice")
            else:
                types.add("unknown")
    return types


def _derive_task_mode(
    intent: str,
    preferred_doc_types: list[str],
    gold_doc_types: set[str],
) -> str:
    """intent + preferred_doc_types + gold_doc_types → task_mode.

    task_mode 종류: sop, ts, issue, general
    """
    pref = set(preferred_doc_types)

    # ts만 선호하고 gold에도 ts 있으면 → ts
    if pref == {"ts"} or (pref <= {"ts"} and "ts" in gold_doc_types and len(gold_doc_types) == 1):
        return "ts"

    # sop/manual 선호 → sop (setup)
    if pref <= {"sop", "manual"} and pref:
        return "sop"

    # troubleshooting이지만 myservice/gcb 포함 → issue
    if intent == "troubleshooting" and gold_doc_types & {"myservice", "gcb"}:
        return "issue"

    # troubleshooting + gold에 sop/ts만 → ts 또는 sop
    if intent == "troubleshooting":
        if "ts" in gold_doc_types:
            return "ts"
        return "sop"

    # procedure → sop
    if intent == "procedure":
        return "sop"

    # information_lookup with myservice/gcb → issue
    if intent == "information_lookup" and gold_doc_types & {"myservice", "gcb"}:
        return "issue"

    return "general"


def _derive_route(task_mode: str, intent: str) -> str:
    """task_mode → expected route.

    route는 LLM 라우터가 정하지만, 기대값으로 사용.
    """
    if task_mode == "sop":
        return "setup"
    if task_mode == "ts":
        return "ts"
    # issue, general → general route
    return "general"


def _derive_eval_type(task_mode: str) -> str:
    """평가 유형 (retrieval 지표 해석에 사용)."""
    if task_mode in ("sop", "ts"):
        return "doc_hit"  # 특정 문서를 찾아야 함
    return "doc_hit"  # issue도 gold_doc_ids hit 기준


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate chat-flow eval dataset")
    parser.add_argument(
        "--input",
        default="data/paper_a/eval/query_gold_master_v0_7_with_implicit.jsonl",
        help="Gold master JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/eval_chatflow_unified.jsonl",
        help="Output JSONL",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Filter by split (dev, test, dev_implicit). Default: all",
    )
    parser.add_argument(
        "--intent-labels",
        default="data/paper_a/train_optional/intent_doctype_labels.jsonl",
        help="Optional intent labels for enrichment (matched by question text)",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    # Load optional intent labels by question text
    intent_by_question: dict[str, dict[str, Any]] = {}
    intent_path = ROOT / args.intent_labels
    if intent_path.exists():
        with intent_path.open(encoding="utf-8") as fp:
            for line in fp:
                row = json.loads(line)
                intent_by_question[row["question"].strip()] = row

    # Load gold master
    with input_path.open(encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp]

    # Filter splits
    if args.splits:
        allowed_splits = set(args.splits)
        rows = [r for r in rows if r.get("split") in allowed_splits]

    # Generate eval dataset
    output_rows: list[dict[str, Any]] = []
    stats: dict[str, Counter] = {
        "task_mode": Counter(),
        "route": Counter(),
        "intent": Counter(),
        "split": Counter(),
    }

    for row in rows:
        q_id = row["q_id"]
        question = row["question"].strip()
        intent = row.get("intent_primary", "")
        preferred = row.get("preferred_doc_types") or []
        gold_doc_ids = row.get("gold_doc_ids") or []

        if not gold_doc_ids:
            continue

        gold_doc_types = _infer_gold_doc_types(gold_doc_ids)

        # Enrich from intent labels if available
        il = intent_by_question.get(question, {})
        if il:
            # Use more detailed intent if available
            if il.get("intent_secondary"):
                intent = il["intent_primary"]
            if il.get("preferred_doc_types"):
                preferred = il["preferred_doc_types"]

        task_mode = _derive_task_mode(intent, preferred, gold_doc_types)
        route = _derive_route(task_mode, intent)

        eval_row: dict[str, Any] = {
            "q_id": q_id,
            "question": question,
            "split": row.get("split", ""),
            "intent_primary": intent,
            "scope_observability": row.get("scope_observability", ""),
            "canonical_device_name": row.get("canonical_device_name", ""),
            "canonical_equip_id": row.get("canonical_equip_id", ""),
            "preferred_doc_types": preferred,
            "gold_doc_ids": gold_doc_ids,
            "gold_doc_types": sorted(gold_doc_types),
            "expected_task_mode": task_mode,
            "expected_route": route,
        }
        output_rows.append(eval_row)

        stats["task_mode"][task_mode] += 1
        stats["route"][route] += 1
        stats["intent"][intent] += 1
        stats["split"][row.get("split", "")] += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for r in output_rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print stats
    print(f"Generated {len(output_rows)} eval rows → {output_path}")
    for key, counter in stats.items():
        print(f"\n{key}:")
        for k, v in counter.most_common():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
