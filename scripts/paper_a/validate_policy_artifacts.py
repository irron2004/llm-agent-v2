from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_KEYS = {"es_doc_id", "es_device_name", "es_doc_type", "es_equip_id", "scope_level", "is_shared"}
VALID_SCOPE_LEVELS = {"shared", "device", "equip"}
EQUIP_DOC_TYPES = {"gcb", "myservice"}


def validate_doc_scope(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_num}: invalid JSON: {e}")
                continue

            if not isinstance(row, dict):
                errors.append(f"line {line_num}: row is not an object")
                continue

            missing = REQUIRED_KEYS - set(row.keys())
            if missing:
                errors.append(f"line {line_num}: missing keys: {sorted(missing)}")
                continue

            scope_level = row.get("scope_level", "")
            if scope_level not in VALID_SCOPE_LEVELS:
                errors.append(
                    f"line {line_num}: invalid scope_level={scope_level!r} "
                    f"(expected one of {sorted(VALID_SCOPE_LEVELS)})"
                )

            is_shared = row.get("is_shared")
            if not isinstance(is_shared, bool):
                errors.append(f"line {line_num}: is_shared must be boolean, got {type(is_shared).__name__}")

            if scope_level == "equip":
                doc_type = str(row.get("es_doc_type", ""))
                if doc_type not in EQUIP_DOC_TYPES:
                    errors.append(
                        f"line {line_num}: scope_level=equip but es_doc_type={doc_type!r} "
                        f"(expected one of {sorted(EQUIP_DOC_TYPES)})"
                    )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Paper A policy artifacts")
    _ = parser.add_argument("--doc-scope", required=True, help="Path to doc_scope.jsonl")
    args = parser.parse_args()

    path = Path(args.doc_scope)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    errors = validate_doc_scope(path)
    if errors:
        print(f"Validation FAILED ({len(errors)} errors):", file=sys.stderr)
        for err in errors[:20]:
            print(f"  {err}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        raise SystemExit(1)

    row_count = sum(1 for line in path.read_text("utf-8").splitlines() if line.strip())
    print(f"Validation passed: {path} ({row_count} rows)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
