from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

V05_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_5.jsonl"
V06_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
OUT_JSONL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_7_mixed.jsonl"
OUT_REPORT_PATH = (
    ROOT / "data/paper_a/eval/query_gold_master_v0_7_mixed_split_report.json"
)
OUT_MD_PATH = (
    ROOT
    / "docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _normalize_v05_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("gold_doc_ids_strict", list(row.get("gold_doc_ids") or []))
    normalized.setdefault("preferred_doc_types", [])
    normalized.setdefault("acceptable_doc_types", [])
    normalized.setdefault("source_doc_id", "")
    normalized.setdefault("source_topic", "")
    normalized.setdefault("retrieved_candidates", [])
    normalized["source"] = (
        f"mixed_from_v0_5_{normalized.get('scope_observability', '')}"
    )
    return normalized


def _leak_key(row: dict[str, Any]) -> str:
    return str(row.get("question_masked") or row.get("question") or "").strip().lower()


def _stable_bucket(key: str, seed: int) -> int:
    payload = f"{key}|{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % 100


def _select_split(bucket: int) -> str:
    return "test" if bucket < 20 else "dev"


def _reassign_splits(
    rows: list[dict[str, Any]], seed: int = 20260314
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _leak_key(row)
        grouped.setdefault(key or str(row.get("q_id") or ""), []).append(dict(row))

    reassigned: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        has_nonempty_gold = any(
            list(row.get("gold_doc_ids") or []) for row in group_rows
        )
        split = (
            "dev" if not has_nonempty_gold else _select_split(_stable_bucket(key, seed))
        )
        for row in group_rows:
            row["split"] = split
            reassigned.append(row)

    scope_nonempty_gold = Counter(
        str(row.get("scope_observability") or "")
        for row in reassigned
        if list(row.get("gold_doc_ids") or [])
    )
    scope_nonempty_gold_test = Counter(
        str(row.get("scope_observability") or "")
        for row in reassigned
        if row.get("split") == "test" and list(row.get("gold_doc_ids") or [])
    )

    for scope, total in scope_nonempty_gold.items():
        min_required = max(1, math.floor(total * 0.1))
        test_count = scope_nonempty_gold_test.get(scope, 0)
        if test_count < min_required:
            raise RuntimeError(
                f"scope={scope}: insufficient test rows after split rebuild ({test_count} < {min_required})"
            )

    return reassigned


def main() -> None:
    v06_rows = _read_jsonl(V06_PATH)
    v05_rows = _read_jsonl(V05_PATH)

    restored_rows = [
        _normalize_v05_row(row)
        for row in v05_rows
        if str(row.get("scope_observability") or "") in {"implicit", "ambiguous"}
    ]
    mixed_rows = _reassign_splits(v06_rows + restored_rows)

    _write_jsonl(OUT_JSONL_PATH, mixed_rows)

    scope_counts = Counter(
        str(row.get("scope_observability") or "") for row in mixed_rows
    )
    split_counts = Counter(str(row.get("split") or "") for row in mixed_rows)
    scope_by_split: dict[str, dict[str, int]] = {}
    empty_gold_counts: dict[str, int] = {}
    leak_dev: set[str] = set()
    leak_test: set[str] = set()

    for scope in sorted(scope_counts):
        scoped_rows = [
            row
            for row in mixed_rows
            if str(row.get("scope_observability") or "") == scope
        ]
        scope_by_split[scope] = dict(
            Counter(str(row.get("split") or "") for row in scoped_rows)
        )
        empty_gold_counts[scope] = sum(
            1 for row in scoped_rows if not list(row.get("gold_doc_ids") or [])
        )

    for row in mixed_rows:
        split = str(row.get("split") or "")
        leak = _leak_key(row)
        if not leak:
            continue
        if split == "dev":
            leak_dev.add(leak)
        elif split == "test":
            leak_test.add(leak)

    report = {
        "date": "2026-03-14",
        "source_paths": {
            "v0_6": str(V06_PATH),
            "v0_5": str(V05_PATH),
        },
        "output_path": str(OUT_JSONL_PATH),
        "total_rows": len(mixed_rows),
        "restored_rows": len(restored_rows),
        "scope_counts": dict(sorted(scope_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "scope_by_split": scope_by_split,
        "empty_gold_counts": empty_gold_counts,
        "overlap_leak_key_count": len(leak_dev & leak_test),
        "overlap_leak_keys_preview": sorted(leak_dev & leak_test)[:20],
    }

    OUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# v0.7 Mixed Eval Restoration (2026-03-14)",
        "",
        "Date: 2026-03-14",
        "Status: generated from `scripts/paper_a/build_v07_mixed_eval_set.py`",
        "",
        "## Inputs",
        "",
        "- Explicit base: `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`",
        "- Restored slices: `data/paper_a/eval/query_gold_master_v0_5.jsonl` (`implicit`, `ambiguous` only)",
        "",
        "## Command",
        "",
        "```bash",
        "cd /home/hskim/work/llm-agent-v2",
        "uv run python scripts/paper_a/build_v07_mixed_eval_set.py",
        "```",
        "",
        "## Summary",
        "",
        f"- Total rows: {len(mixed_rows)}",
        f"- Restored rows from v0.5: {len(restored_rows)}",
        f"- Scope counts: {dict(sorted(scope_counts.items()))}",
        f"- Split counts: {dict(sorted(split_counts.items()))}",
        f"- Dev/Test leak-key overlap: {len(leak_dev & leak_test)}",
        "",
        "## Scope by split",
        "",
    ]

    for scope, counts in sorted(scope_by_split.items()):
        lines.append(f"- {scope}: {counts}")

    lines.extend(
        [
            "",
            "## Empty gold counts",
            "",
        ]
    )
    for scope, count in sorted(empty_gold_counts.items()):
        lines.append(f"- {scope}: {count}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v0.7 mixed restores the missing implicit/ambiguous slices while preserving the stronger v0.6 explicit set as the base.",
            "- ambiguous rows still carry empty-gold limitations from v0.5 and should be reported separately from gold-bearing slices.",
            "- The merged file is intended to unblock mixed-scope reporting and future experiment runs, not to hide the coverage limits of ambiguous rows.",
        ]
    )

    OUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved mixed eval set to: {OUT_JSONL_PATH}")
    print(f"Saved split report to: {OUT_REPORT_PATH}")
    print(f"Saved mixed eval evidence to: {OUT_MD_PATH}")


if __name__ == "__main__":
    main()
