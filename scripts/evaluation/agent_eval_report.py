#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


def _normalize_doc_id(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\.(pdf|docx|doc|txt)$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _parse_page_range(raw: object) -> tuple[int, int] | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if "-" in value:
        left, right = value.split("-", 1)
        try:
            a = int(left.strip())
            b = int(right.strip())
        except ValueError:
            return None
        if a <= b:
            return (a, b)
        return (b, a)
    try:
        page = int(value)
    except ValueError:
        return None
    return (page, page)


def _as_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    source = cast(dict[object, object], value)
    out: dict[str, object] = {}
    for key, item in source.items():
        out[str(key)] = item
    return out


def _extract_page(doc: Mapping[str, object]) -> int | None:
    page_raw = doc.get("page")
    if isinstance(page_raw, int):
        return page_raw
    if isinstance(page_raw, str):
        text = page_raw.strip()
        if text.isdigit():
            return int(text)
    return None


def _in_expected_range(page: int | None, expected: tuple[int, int] | None) -> bool:
    if page is None or expected is None:
        return False
    return expected[0] <= page <= expected[1]


@dataclass
class EvalItem:
    idx: int
    expected_doc_norm: str
    expected_page_range: tuple[int, int] | None
    retrieved_docs: list[dict[str, object]]
    has_error: bool


@dataclass
class Metrics:
    total: int
    failures: int
    doc_hit_10: int
    page_hit_1: int
    page_hit_3: int
    page_hit_5: int
    first_page_1: int
    top10_docs_by_idx: dict[int, set[str]]


@dataclass
class CliArgs:
    before_jsonl: Path
    after_jsonl: Path
    out: Path


def _load_jsonl(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = cast(object, json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            payload_map = _as_mapping(payload)
            if payload_map is None:
                raise RuntimeError(
                    f"Invalid row type at {path}:{line_no}: object expected"
                )

            idx_raw = payload_map.get("idx")
            if isinstance(idx_raw, bool):
                raise RuntimeError(f"Invalid idx at {path}:{line_no}: {idx_raw!r}")
            if isinstance(idx_raw, int):
                idx = idx_raw
            elif isinstance(idx_raw, float):
                idx = int(idx_raw)
            elif isinstance(idx_raw, str):
                text = idx_raw.strip()
                try:
                    idx = int(text)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid idx at {path}:{line_no}: {idx_raw!r}"
                    ) from exc
            else:
                raise RuntimeError(f"Invalid idx at {path}:{line_no}: {idx_raw!r}")

            expected_doc = str(payload_map.get("expected_doc") or "")
            expected_doc_norm = _normalize_doc_id(expected_doc)

            expected_pages = payload_map.get("expected_pages")
            expected_page_range = _parse_page_range(expected_pages)

            response_map = _as_mapping(payload_map.get("response"))
            retrieved_raw: object = []
            if response_map is not None:
                retrieved_raw = response_map.get("retrieved_docs", [])

            retrieved_docs: list[dict[str, object]] = []
            if isinstance(retrieved_raw, list):
                retrieved_list = cast(list[object], retrieved_raw)
                for entry in retrieved_list:
                    entry_map = _as_mapping(entry)
                    if entry_map is not None:
                        retrieved_docs.append(entry_map)

            error_text = str(payload_map.get("error") or "").strip()
            items.append(
                EvalItem(
                    idx=idx,
                    expected_doc_norm=expected_doc_norm,
                    expected_page_range=expected_page_range,
                    retrieved_docs=retrieved_docs,
                    has_error=bool(error_text),
                )
            )
    if not items:
        raise RuntimeError(f"No valid JSONL rows found in {path}")
    return items


def _has_doc_hit(
    top_docs: list[dict[str, object]], expected_doc_norm: str, k: int
) -> bool:
    for doc in top_docs[:k]:
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id == expected_doc_norm:
            return True
    return False


def _has_page_hit(
    top_docs: list[dict[str, object]],
    expected_doc_norm: str,
    expected_range: tuple[int, int] | None,
    k: int,
) -> bool:
    for doc in top_docs[:k]:
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id != expected_doc_norm:
            continue
        if _in_expected_range(_extract_page(doc), expected_range):
            return True
    return False


def _is_first_page_one(top_docs: list[dict[str, object]]) -> bool:
    if not top_docs:
        return False
    first_page = top_docs[0].get("page")
    if isinstance(first_page, int):
        return first_page == 1
    if isinstance(first_page, str):
        text = first_page.strip()
        return text.isdigit() and int(text) == 1
    return False


def _compute_metrics(items: list[EvalItem]) -> Metrics:
    total = len(items)
    failures = 0
    doc_hit_10 = 0
    page_hit_1 = 0
    page_hit_3 = 0
    page_hit_5 = 0
    first_page_1 = 0
    top10_docs_by_idx: dict[int, set[str]] = {}

    for item in items:
        if item.has_error:
            failures += 1

        if _has_doc_hit(item.retrieved_docs, item.expected_doc_norm, 10):
            doc_hit_10 += 1
        if _has_page_hit(
            item.retrieved_docs, item.expected_doc_norm, item.expected_page_range, 1
        ):
            page_hit_1 += 1
        if _has_page_hit(
            item.retrieved_docs, item.expected_doc_norm, item.expected_page_range, 3
        ):
            page_hit_3 += 1
        if _has_page_hit(
            item.retrieved_docs, item.expected_doc_norm, item.expected_page_range, 5
        ):
            page_hit_5 += 1
        if _is_first_page_one(item.retrieved_docs):
            first_page_1 += 1

        top10: set[str] = set()
        for doc in item.retrieved_docs[:10]:
            norm = _normalize_doc_id(str(doc.get("id") or ""))
            if norm:
                top10.add(norm)
        top10_docs_by_idx[item.idx] = top10

    return Metrics(
        total=total,
        failures=failures,
        doc_hit_10=doc_hit_10,
        page_hit_1=page_hit_1,
        page_hit_3=page_hit_3,
        page_hit_5=page_hit_5,
        first_page_1=first_page_1,
        top10_docs_by_idx=top10_docs_by_idx,
    )


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _count_cell(count: int, total: int) -> str:
    return f"{count}/{total} ({_ratio(count, total) * 100:.2f}%)"


def _delta_count_ratio(
    before_count: int, before_total: int, after_count: int, after_total: int
) -> str:
    delta_count = after_count - before_count
    delta_ratio_pp = (
        _ratio(after_count, after_total) - _ratio(before_count, before_total)
    ) * 100
    return f"{delta_count:+d} ({delta_ratio_pp:+.2f}pp)"


def _mean_jaccard_10(before: Metrics, after: Metrics) -> tuple[float | None, int]:
    pairs = sorted(set(before.top10_docs_by_idx) & set(after.top10_docs_by_idx))
    if not pairs:
        return None, 0

    total = 0.0
    for idx in pairs:
        left = before.top10_docs_by_idx[idx]
        right = after.top10_docs_by_idx[idx]
        union = left | right
        if not union:
            total += 1.0
            continue
        total += len(left & right) / len(union)
    return total / len(pairs), len(pairs)


def _format_jaccard(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _build_report(
    before: Metrics, after: Metrics, before_path: Path, after_path: Path
) -> str:
    jaccard_mean, jaccard_n = _mean_jaccard_10(before, after)

    lines: list[str] = [
        "# Agent Eval Before/After Report",
        "",
        f"- before_jsonl: `{before_path}`",
        f"- after_jsonl: `{after_path}`",
        f"- total(before/after): {before.total}/{after.total}",
        "",
        "| KPI | BEFORE | AFTER | DELTA |",
        "|---|---:|---:|---:|",
        f"| doc-hit@10 | {_count_cell(before.doc_hit_10, before.total)} | {_count_cell(after.doc_hit_10, after.total)} | {_delta_count_ratio(before.doc_hit_10, before.total, after.doc_hit_10, after.total)} |",
        f"| page-hit@1 | {_count_cell(before.page_hit_1, before.total)} | {_count_cell(after.page_hit_1, after.total)} | {_delta_count_ratio(before.page_hit_1, before.total, after.page_hit_1, after.total)} |",
        f"| page-hit@3 | {_count_cell(before.page_hit_3, before.total)} | {_count_cell(after.page_hit_3, after.total)} | {_delta_count_ratio(before.page_hit_3, before.total, after.page_hit_3, after.total)} |",
        f"| page-hit@5 | {_count_cell(before.page_hit_5, before.total)} | {_count_cell(after.page_hit_5, after.total)} | {_delta_count_ratio(before.page_hit_5, before.total, after.page_hit_5, after.total)} |",
        f"| first_page=1 | {_count_cell(before.first_page_1, before.total)} | {_count_cell(after.first_page_1, after.total)} | {_delta_count_ratio(before.first_page_1, before.total, after.first_page_1, after.total)} |",
        f"| failures | {_count_cell(before.failures, before.total)} | {_count_cell(after.failures, after.total)} | {_delta_count_ratio(before.failures, before.total, after.failures, after.total)} |",
        f"| mean-jaccard@10 | - | {_format_jaccard(jaccard_mean)} | {'-' if jaccard_mean is None else 'paired by idx'} |",
        "",
        "- mean-jaccard@10 pairs: "
        + ("0 (no overlapping idx)" if jaccard_mean is None else f"{jaccard_n}"),
        "",
        "Metrics footer:",
        "- doc-id normalization: lower-case, strip .pdf/.docx/.doc/.txt, non-alnum->underscore, collapse underscores.",
        "- page range parsing: accepts empty, single int (`7`), or range (`6-14`); reversed range is normalized.",
        "- doc-hit@10: expected doc id appears in top 10 retrieved docs.",
        "- page-hit@1/@3/@5: within top K, expected doc appears and retrieved page falls in expected page range.",
        "- first_page=1: first retrieved doc page equals 1 (int or digit string).",
        "- failures: rows where `error` is non-empty.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate BEFORE/AFTER KPI markdown from evaluator JSONL files."
    )
    _ = parser.add_argument("--before-jsonl", required=True)
    _ = parser.add_argument("--after-jsonl", required=True)
    _ = parser.add_argument("--out", required=True)

    parsed_raw = dict(vars(parser.parse_args()))
    before_jsonl = Path(str(parsed_raw.get("before_jsonl") or ""))
    after_jsonl = Path(str(parsed_raw.get("after_jsonl") or ""))
    out = Path(str(parsed_raw.get("out") or ""))
    return CliArgs(before_jsonl=before_jsonl, after_jsonl=after_jsonl, out=out)


def main() -> int:
    args = _parse_args()

    before_items = _load_jsonl(args.before_jsonl)
    after_items = _load_jsonl(args.after_jsonl)

    before_metrics = _compute_metrics(before_items)
    after_metrics = _compute_metrics(after_items)

    report = _build_report(
        before_metrics, after_metrics, args.before_jsonl, args.after_jsonl
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _ = args.out.write_text(report, encoding="utf-8")

    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        raise SystemExit(code)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
