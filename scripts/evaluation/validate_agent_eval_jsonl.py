#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _coerce_str_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    mapped: dict[str, Any] = {}
    for key, item in value.items():
        mapped[str(key)] = item
    return mapped


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _has_legacy_retry_fields(row: dict[str, Any]) -> bool:
    if "attempts" in row or "retry_count" in row:
        return True
    metadata = _coerce_str_mapping(row.get("metadata"))
    if metadata is not None and ("attempts" in metadata or "retry_count" in metadata):
        return True
    response = _coerce_str_mapping(row.get("response"))
    if response is None:
        return False
    if "attempts" in response or "retry_count" in response:
        return True
    response_metadata = _coerce_str_mapping(response.get("metadata"))
    if response_metadata is None:
        return False
    return "attempts" in response_metadata or "retry_count" in response_metadata


def _resolve_retry_count(row: dict[str, Any]) -> int | None:
    trace = _coerce_str_mapping(row.get("trace"))
    if trace is not None and "retry_count" in trace:
        return _parse_int(trace.get("retry_count"))

    metadata = _coerce_str_mapping(row.get("metadata"))
    if metadata is not None:
        if "attempts" in metadata:
            return _parse_int(metadata.get("attempts"))
        if "retry_count" in metadata:
            return _parse_int(metadata.get("retry_count"))

    response = _coerce_str_mapping(row.get("response"))
    if response is not None:
        response_trace = _coerce_str_mapping(response.get("trace"))
        if response_trace is not None and "retry_count" in response_trace:
            return _parse_int(response_trace.get("retry_count"))

        response_metadata = _coerce_str_mapping(response.get("metadata"))
        if response_metadata is not None:
            if "attempts" in response_metadata:
                return _parse_int(response_metadata.get("attempts"))
            if "retry_count" in response_metadata:
                return _parse_int(response_metadata.get("retry_count"))

        if "attempts" in response:
            return _parse_int(response.get("attempts"))
        if "retry_count" in response:
            return _parse_int(response.get("retry_count"))

    if "attempts" in row:
        return _parse_int(row.get("attempts"))
    if "retry_count" in row:
        return _parse_int(row.get("retry_count"))
    return None


def _resolve_answer(row: dict[str, Any]) -> str | None:
    answer = row.get("answer")
    if isinstance(answer, str):
        return answer
    if answer is None:
        response = _coerce_str_mapping(row.get("response"))
        if response is not None and isinstance(response.get("answer"), str):
            return str(response.get("answer"))
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical agent evaluation JSONL schema"
    )
    _ = parser.add_argument("--jsonl", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.jsonl)
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")

    total = 0
    invalid = 0
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            total += 1
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
                invalid += 1
                continue
            row = _coerce_str_mapping(parsed)
            if row is None:
                print(f"{path}:{line_no}: JSON object expected", file=sys.stderr)
                invalid += 1
                continue

            answer = _resolve_answer(row)
            if answer is None:
                print(
                    f"{path}:{line_no}: missing canonical top-level answer (string)",
                    file=sys.stderr,
                )
                invalid += 1

            retry_count = _resolve_retry_count(row)
            if retry_count is None:
                print(
                    f"{path}:{line_no}: missing trace.retry_count (int)",
                    file=sys.stderr,
                )
                invalid += 1
            else:
                trace = _coerce_str_mapping(row.get("trace"))
                if trace is None or not isinstance(trace.get("retry_count"), int):
                    print(
                        f"{path}:{line_no}: canonical trace.retry_count must exist as int",
                        file=sys.stderr,
                    )
                    invalid += 1

            if _has_legacy_retry_fields(row):
                print(
                    f"{path}:{line_no}: legacy retry fields detected (attempts/retry_count outside trace)",
                    file=sys.stderr,
                )
                invalid += 1

    if total == 0:
        raise RuntimeError(f"No JSONL rows found: {path}")
    if invalid > 0:
        print(
            json.dumps(
                {"ok": False, "rows": total, "invalid": invalid}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps({"ok": True, "rows": total, "invalid": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
