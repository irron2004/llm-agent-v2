from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Paper A eval-set JSONL")
    _ = parser.add_argument("--path", required=True, help="Path to eval-set JSONL")
    return parser.parse_args()


def _is_list_of_strings(value: object) -> bool:
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    return all(isinstance(item, str) for item in items)


def run() -> int:
    args = parse_args()
    path = Path(cast(str, args.path))

    if not path.exists() or not path.is_file():
        print(f"JSONL file not found: {path}", file=sys.stderr)
        return 1

    required_keys = [
        "qid",
        "split",
        "query",
        "target_device",
        "gold_doc_ids",
        "gold_source_files",
        "gold_pages",
    ]

    try:
        with path.open("r", encoding="utf-8") as f:
            row_count = 0
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                row_count += 1

                try:
                    loaded = cast(object, json.loads(raw))
                except json.JSONDecodeError as exc:
                    print(f"Line {line_no}: invalid JSON ({exc})", file=sys.stderr)
                    return 1

                if not isinstance(loaded, dict):
                    print(f"Line {line_no}: row must be a JSON object", file=sys.stderr)
                    return 1
                row = cast(dict[str, object], loaded)

                for key in required_keys:
                    if key not in row:
                        print(
                            f"Line {line_no}: missing required key '{key}'",
                            file=sys.stderr,
                        )
                        return 1

                if not isinstance(row["qid"], str) or not row["qid"].strip():
                    print(
                        f"Line {line_no}: 'qid' must be a non-empty string",
                        file=sys.stderr,
                    )
                    return 1
                if not isinstance(row["split"], str) or not row["split"].strip():
                    print(
                        f"Line {line_no}: 'split' must be a non-empty string",
                        file=sys.stderr,
                    )
                    return 1
                if not isinstance(row["query"], str):
                    print(f"Line {line_no}: 'query' must be a string", file=sys.stderr)
                    return 1
                if not isinstance(row["target_device"], str):
                    print(
                        f"Line {line_no}: 'target_device' must be a string",
                        file=sys.stderr,
                    )
                    return 1
                if not _is_list_of_strings(row["gold_doc_ids"]):
                    print(
                        f"Line {line_no}: 'gold_doc_ids' must be a list of strings",
                        file=sys.stderr,
                    )
                    return 1
                gold_doc_ids = cast(list[str], row["gold_doc_ids"])
                if len(gold_doc_ids) == 0:
                    print(
                        f"Line {line_no}: 'gold_doc_ids' must contain at least one item",
                        file=sys.stderr,
                    )
                    return 1
                if not _is_list_of_strings(row["gold_source_files"]):
                    print(
                        f"Line {line_no}: 'gold_source_files' must be a list of strings",
                        file=sys.stderr,
                    )
                    return 1
                if not isinstance(row["gold_pages"], list):
                    print(
                        f"Line {line_no}: 'gold_pages' must be a list", file=sys.stderr
                    )
                    return 1
                if "topic" in row and not isinstance(row["topic"], str):
                    print(f"Line {line_no}: 'topic' must be a string", file=sys.stderr)
                    return 1

            if row_count == 0:
                print("No rows found in JSONL", file=sys.stderr)
                return 1
    except Exception as exc:
        print(f"validate_eval_jsonl failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validation passed: {path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
