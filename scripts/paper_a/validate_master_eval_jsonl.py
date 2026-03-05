from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast


REQUIRED_KEYS = [
    "q_id",
    "split",
    "question",
    "scope_observability",
    "intent_primary",
    "target_scope_level",
    "allowed_devices",
    "allowed_equips",
    "shared_allowed",
    "family_allowed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Paper A master eval-set JSONL"
    )
    _ = parser.add_argument("--path", required=True, help="Path to eval-set JSONL")
    _ = parser.add_argument(
        "--require-gold",
        action="store_true",
        help="Fail if selected rows contain empty gold_doc_ids",
    )
    _ = parser.add_argument(
        "--scope-observability",
        default="all",
        help="Filter by scope_observability (all|explicit_device|explicit_equip|implicit|ambiguous)",
    )
    _ = parser.add_argument(
        "--split",
        default="all",
        help="Filter by split (all|dev|test)",
    )
    _ = parser.add_argument(
        "--report-path",
        default=".sisyphus/evidence/task-02-master-validator-report.json",
        help="Output path for validation report",
    )
    return parser.parse_args()


def _is_list_of_strings(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in cast(list[object], value))


def _normalize_selector(value: str) -> str:
    return str(value or "").strip().lower()


def run() -> int:
    args = parse_args()
    path = Path(cast(str, args.path))
    report_path = Path(cast(str, args.report_path))
    selected_scope = _normalize_selector(cast(str, args.scope_observability))
    selected_split = _normalize_selector(cast(str, args.split))

    if not path.exists() or not path.is_file():
        print(f"JSONL file not found: {path}", file=sys.stderr)
        return 1

    if selected_scope not in {
        "all",
        "explicit_device",
        "explicit_equip",
        "implicit",
        "ambiguous",
    }:
        print(
            f"Invalid --scope-observability: {selected_scope}",
            file=sys.stderr,
        )
        return 1

    if selected_split not in {"all", "dev", "test"}:
        print(f"Invalid --split: {selected_split}", file=sys.stderr)
        return 1

    row_count = 0
    selected_count = 0
    empty_gold: list[str] = []
    parse_errors: list[str] = []
    schema_errors: list[str] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                row_count += 1

                try:
                    loaded = cast(object, json.loads(raw))
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"line {line_no}: invalid JSON ({exc})")
                    continue

                if not isinstance(loaded, dict):
                    schema_errors.append(f"line {line_no}: row must be a JSON object")
                    continue

                row = cast(dict[str, object], loaded)

                for key in REQUIRED_KEYS:
                    if key not in row:
                        schema_errors.append(
                            f"line {line_no}: missing required key '{key}'"
                        )

                if (
                    not isinstance(row.get("q_id"), str)
                    or not str(row.get("q_id")).strip()
                ):
                    schema_errors.append(
                        f"line {line_no}: q_id must be a non-empty string"
                    )

                split = str(row.get("split") or "").strip().lower()
                scope = str(row.get("scope_observability") or "").strip().lower()

                if not split:
                    schema_errors.append(
                        f"line {line_no}: split must be a non-empty string"
                    )
                if not scope:
                    schema_errors.append(
                        f"line {line_no}: scope_observability must be a non-empty string"
                    )

                if not isinstance(row.get("question"), str):
                    schema_errors.append(f"line {line_no}: question must be a string")

                if not _is_list_of_strings(row.get("allowed_devices")):
                    schema_errors.append(
                        f"line {line_no}: allowed_devices must be a list of strings"
                    )

                if not _is_list_of_strings(row.get("allowed_equips")):
                    schema_errors.append(
                        f"line {line_no}: allowed_equips must be a list of strings"
                    )

                if not isinstance(row.get("shared_allowed"), bool):
                    schema_errors.append(
                        f"line {line_no}: shared_allowed must be a boolean"
                    )

                if not isinstance(row.get("family_allowed"), bool):
                    schema_errors.append(
                        f"line {line_no}: family_allowed must be a boolean"
                    )

                if not isinstance(row.get("gold_doc_ids", []), list):
                    schema_errors.append(f"line {line_no}: gold_doc_ids must be a list")

                if selected_split != "all" and split != selected_split:
                    continue
                if selected_scope != "all" and scope != selected_scope:
                    continue

                selected_count += 1
                if bool(args.require_gold):
                    gold_doc_ids = row.get("gold_doc_ids", [])
                    if not isinstance(gold_doc_ids, list) or len(gold_doc_ids) == 0:
                        qid = str(row.get("q_id") or "")
                        empty_gold.append(qid)
    except OSError as exc:
        print(f"Failed to read JSONL: {exc}", file=sys.stderr)
        return 1

    report = {
        "path": str(path),
        "total_rows": row_count,
        "selected_rows": selected_count,
        "scope_observability": selected_scope,
        "split": selected_split,
        "require_gold": bool(args.require_gold),
        "parse_errors": parse_errors,
        "schema_errors": schema_errors,
        "empty_gold_qids": empty_gold,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2, sort_keys=True)
        _ = rf.write("\n")

    if row_count == 0:
        print("No rows found in JSONL", file=sys.stderr)
        return 1

    if parse_errors:
        print(parse_errors[0], file=sys.stderr)
        return 1

    if schema_errors:
        print(schema_errors[0], file=sys.stderr)
        return 1

    if bool(args.require_gold) and empty_gold:
        print(
            (
                "Empty gold_doc_ids detected for selected rows. "
                f"Count={len(empty_gold)}. See {report_path}"
            ),
            file=sys.stderr,
        )
        return 1

    print(f"Validation passed: {path} (selected_rows={selected_count})")
    print(f"Report written: {report_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
