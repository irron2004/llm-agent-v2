#!/usr/bin/env python3
"""Validate SOP filter evaluation JSONL artifacts (thin + rich)."""

from __future__ import annotations

import argparse
import json
import re
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


def _is_rich_row(row: dict[str, Any]) -> bool:
    """Detect if this is a rich row by presence of rich-specific fields."""
    return "top_docs" in row or "request_payload" in row


def _validate_schema_version(
    row: dict[str, Any], line_no: int, path: Path
) -> list[str]:
    """Validate schema_version is present and non-empty."""
    errors = []
    schema_version = row.get("schema_version")
    if schema_version is None:
        errors.append(f"{path}:{line_no}: missing schema_version")
    elif not isinstance(schema_version, str) or not schema_version.strip():
        errors.append(f"{path}:{line_no}: schema_version must be non-empty string")
    return errors


def _validate_required_keys(
    row: dict[str, Any], line_no: int, path: Path, *, is_rich: bool
) -> list[str]:
    """Validate required top-level keys for both thin and rich rows."""
    errors = []
    required: list[str] = [
        "idx",
        "question",
        "filter",
        "filter_doc_types",
        "gold_doc",
        "gold_pages",
        "hit_doc",
        "hit_page",
        "hit_rank",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "match_debug",
        "elapsed_ms",
        "error",
    ]
    if is_rich:
        required.extend(["request_payload", "response_metadata", "top_docs", "answer"])
    for key in required:
        if key not in row:
            errors.append(f"{path}:{line_no}: missing required key '{key}'")
    return errors


def _validate_types(
    row: dict[str, Any], line_no: int, path: Path, *, is_rich: bool
) -> list[str]:
    errors: list[str] = []
    if not isinstance(row.get("idx"), int):
        errors.append(f"{path}:{line_no}: idx must be int")
    if not isinstance(row.get("question"), str):
        errors.append(f"{path}:{line_no}: question must be str")
    if not isinstance(row.get("filter"), str):
        errors.append(f"{path}:{line_no}: filter must be str")

    filter_doc_types = row.get("filter_doc_types")
    if not isinstance(filter_doc_types, list) or not all(
        isinstance(x, str) for x in filter_doc_types
    ):
        errors.append(f"{path}:{line_no}: filter_doc_types must be list[str]")

    for key in ("gold_doc", "gold_pages"):
        if not isinstance(row.get(key), str):
            errors.append(f"{path}:{line_no}: {key} must be str")

    for key in (
        "hit_doc",
        "hit_page",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
    ):
        if not isinstance(row.get(key), bool):
            errors.append(f"{path}:{line_no}: {key} must be bool")

    hit_rank = row.get("hit_rank")
    if hit_rank is not None and not isinstance(hit_rank, int):
        errors.append(f"{path}:{line_no}: hit_rank must be int or null")

    if not isinstance(row.get("match_debug"), dict):
        errors.append(f"{path}:{line_no}: match_debug must be dict")

    if not isinstance(row.get("elapsed_ms"), (int, float)):
        errors.append(f"{path}:{line_no}: elapsed_ms must be number")

    err = row.get("error")
    if err is not None and not isinstance(err, str):
        errors.append(f"{path}:{line_no}: error must be str or null")

    if is_rich:
        if not isinstance(row.get("request_payload"), dict):
            errors.append(f"{path}:{line_no}: request_payload must be dict")
        if not isinstance(row.get("response_metadata"), dict):
            errors.append(f"{path}:{line_no}: response_metadata must be dict")
        if not isinstance(row.get("top_docs"), list):
            errors.append(f"{path}:{line_no}: top_docs must be list")
        if not isinstance(row.get("answer"), str):
            errors.append(f"{path}:{line_no}: answer must be str")
    return errors


def _validate_rich_fields(row: dict[str, Any], line_no: int, path: Path) -> list[str]:
    """Validate rich-specific fields when present."""
    errors = []

    # Check top_docs is a list if present
    if "top_docs" in row:
        top_docs = row["top_docs"]
        if not isinstance(top_docs, list):
            errors.append(
                f"{path}:{line_no}: top_docs must be a list, got {type(top_docs).__name__}"
            )

    # Check answer is a string if present
    if "answer" in row:
        answer = row["answer"]
        if not isinstance(answer, str):
            errors.append(
                f"{path}:{line_no}: answer must be a string, got {type(answer).__name__}"
            )

    # request_payload should exist for rich rows
    if "request_payload" not in row:
        errors.append(f"{path}:{line_no}: rich row missing request_payload")

    return errors


def _check_references_ok(answer: str, has_top_docs: bool) -> bool:
    """Check if answer contains reference section when docs are available."""
    if not has_top_docs:
        return True  # Skip if no docs to reference

    # Look for common reference section markers
    ref_markers = ["참고문헌", "References", "참고자료", "References:", "참고"]
    answer_lower = answer.lower()
    return any(marker.lower() in answer_lower for marker in ref_markers)


def _check_citations_ok(answer: str, has_top_docs: bool) -> bool:
    """Check if answer contains citation tokens when docs are available."""
    if not has_top_docs:
        return True  # Skip if no docs to cite

    # Look for citation patterns like [1], [2], [12], etc.
    citation_pattern = re.compile(r"\[[0-9]+\]")
    return bool(citation_pattern.search(answer))


def _check_language_ok(row: dict[str, Any]) -> bool:
    """Check if response language matches target_language from metadata or question."""
    # Try to get target_language from response_metadata
    response_metadata = _coerce_str_mapping(row.get("response_metadata"))
    target_lang = None

    if response_metadata:
        target_lang = response_metadata.get("target_language")

    # Fallback: check question/answer for language hints
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))

    # If we have target_language from metadata, use it
    if target_lang:
        if target_lang == "ko":
            # Korean: check for Hangul presence
            has_hangul = bool(re.search(r"[가-힣]", question + answer))
            return has_hangul
        elif target_lang == "en":
            # English: check for ASCII dominance (less than 20% non-ASCII)
            combined = question + answer
            if len(combined) == 0:
                return True
            non_ascii = sum(1 for c in combined if ord(c) > 127)
            return non_ascii / len(combined) < 0.2

    # No target_language: infer from content
    # If answer contains Hangul, assume Korean
    if re.search(r"[가-힣]", answer):
        return True
    # If answer is mostly ASCII, assume English
    if len(answer) > 0:
        non_ascii = sum(1 for c in answer if ord(c) > 127)
        return non_ascii / len(answer) < 0.2
    # Default to true if no answer yet
    return True


def _validate_answer_checks(
    row: dict[str, Any], line_no: int, path: Path
) -> dict[str, bool]:
    """Run answer content checks. Returns dict of check results."""
    top_docs = row.get("top_docs")
    has_top_docs = isinstance(top_docs, list) and len(top_docs) > 0

    answer = str(row.get("answer", ""))

    return {
        "references_ok": _check_references_ok(answer, has_top_docs),
        "citations_ok": _check_citations_ok(answer, has_top_docs),
        "language_ok": _check_language_ok(row),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SOP filter evaluation JSONL (thin + rich)"
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
    rich_count = 0
    thin_count = 0

    # Track validation results per row
    results: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue

            total += 1
            row_errors: list[str] = []
            row_checks: dict[str, bool] = {}

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
                invalid += 1
                results.append(
                    {"line": line_no, "valid": False, "errors": ["invalid JSON"]}
                )
                continue

            row = _coerce_str_mapping(parsed)
            if row is None:
                print(f"{path}:{line_no}: JSON object expected", file=sys.stderr)
                invalid += 1
                results.append(
                    {"line": line_no, "valid": False, "errors": ["not a JSON object"]}
                )
                continue

            # Check if rich or thin
            is_rich = _is_rich_row(row)
            if is_rich:
                rich_count += 1
            else:
                thin_count += 1

            # Run validations
            row_errors.extend(_validate_schema_version(row, line_no, path))
            row_errors.extend(
                _validate_required_keys(row, line_no, path, is_rich=is_rich)
            )
            row_errors.extend(_validate_types(row, line_no, path, is_rich=is_rich))

            if is_rich:
                row_errors.extend(_validate_rich_fields(row, line_no, path))

            # Validate answer is not empty for rich rows without errors
            if is_rich:
                has_error = (
                    row.get("error") is not None and str(row.get("error")).strip()
                )
                answer = str(row.get("answer", "")).strip()
                if not has_error and not answer:
                    row_errors.append(
                        f"{path}:{line_no}: rich row has no error but answer is empty"
                    )

            # Run answer checks for rich rows
            if is_rich:
                row_checks = _validate_answer_checks(row, line_no, path)

            if row_errors:
                invalid += 1
                for err in row_errors:
                    print(err, file=sys.stderr)

            results.append(
                {
                    "line": line_no,
                    "valid": len(row_errors) == 0,
                    "errors": row_errors,
                    "checks": row_checks,
                    "is_rich": is_rich,
                    "has_error": bool(
                        row.get("error") is not None and str(row.get("error")).strip()
                    ),
                }
            )

    if total == 0:
        raise RuntimeError(f"No JSONL rows found: {path}")

    rich_results = [r for r in results if bool(r.get("is_rich"))]

    ref_ok_count = sum(
        1 for r in rich_results if r.get("checks", {}).get("references_ok", True)
    )
    cit_ok_count = sum(
        1 for r in rich_results if r.get("checks", {}).get("citations_ok", True)
    )
    lang_ok_count = sum(
        1 for r in rich_results if r.get("checks", {}).get("language_ok", True)
    )
    format_ok_count = sum(
        1
        for r in rich_results
        if r.get("checks", {}).get("references_ok", True)
        and r.get("checks", {}).get("citations_ok", True)
        and r.get("checks", {}).get("language_ok", True)
    )

    # Build summary
    error_count = sum(1 for r in results if bool(r.get("has_error")))
    summary = {
        "ok": invalid == 0,
        "rows": total,
        "invalid": invalid,
        "thin_rows": thin_count,
        "rich_rows": rich_count,
        "errors": error_count,
    }  # type: dict[str, Any]

    if rich_count > 0:
        summary["checks"] = {
            "references_ok": f"{ref_ok_count}/{rich_count}",
            "citations_ok": f"{cit_ok_count}/{rich_count}",
            "language_ok": f"{lang_ok_count}/{rich_count}",
            "format_ok": f"{format_ok_count}/{rich_count}",
        }

    if invalid > 0:
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
