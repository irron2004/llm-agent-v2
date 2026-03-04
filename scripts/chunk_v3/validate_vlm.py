"""Phase 1 검증: VLM 파싱 결과 품질 검사.

PDF/PPTX 원본 페이지 수 vs VLM 파싱 페이지 수 비교,
빈 페이지 감지, 중복 페이지 감지, 커버리지 리포트.

Usage:
    python scripts/chunk_v3/validate_vlm.py \
        --parsed-dir data/vlm_parsed \
        --source-dir /home/llm-share/datasets/pe_agent_data/pe_preprocess_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def get_pdf_page_count(pdf_path: Path) -> int | None:
    """pdfinfo로 PDF 페이지 수 추출."""
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def get_pptx_slide_count(pptx_path: Path) -> int | None:
    """zipfile로 PPTX 슬라이드 수 추출."""
    import zipfile
    try:
        with zipfile.ZipFile(pptx_path, "r") as zf:
            slides = [n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            return len(slides)
    except Exception:
        pass
    return None


def get_expected_pages(source_path: Path) -> int | None:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return get_pdf_page_count(source_path)
    elif suffix == ".pptx":
        return get_pptx_slide_count(source_path)
    return None


def build_source_map(source_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for f in source_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".pdf", ".pptx"):
            mapping[f.name] = f
    return mapping


def validate_single(parsed_json: dict[str, Any], source_path: Path | None) -> dict[str, Any]:
    doc_id = parsed_json.get("doc_id", "unknown")
    source_file = parsed_json.get("source_file", "")
    pages = parsed_json.get("pages", [])
    parsed_count = len(pages)

    expected = None
    if source_path and source_path.exists():
        expected = get_expected_pages(source_path)

    parsed_page_nums = {p["page"] for p in pages}
    missing_pages: list[int] = []
    if expected:
        for i in range(1, expected + 1):
            if i not in parsed_page_nums:
                missing_pages.append(i)

    empty_pages: list[int] = []
    short_pages: list[int] = []
    char_counts: list[int] = []

    for p in pages:
        text = p.get("text", "")
        char_count = len(text.strip())
        char_counts.append(char_count)
        if char_count < 50:
            empty_pages.append(p["page"])
        elif char_count < 150:
            short_pages.append(p["page"])

    duplicate_pages: list[tuple[int, int]] = []
    page_hashes: list[tuple[int, str]] = []
    for p in sorted(pages, key=lambda x: x["page"]):
        text = p.get("text", "").strip()
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        page_hashes.append((p["page"], h))

    for i in range(1, len(page_hashes)):
        if page_hashes[i][1] == page_hashes[i - 1][1] and len(pages[i].get("text", "").strip()) > 10:
            duplicate_pages.append((page_hashes[i - 1][0], page_hashes[i][0]))

    coverage = None
    if expected and expected > 0:
        coverage = round(parsed_count / expected, 4)

    page_match = None
    if expected is not None:
        page_match = (parsed_count == expected)

    avg_chars = round(sum(char_counts) / len(char_counts), 1) if char_counts else 0.0

    issues: list[str] = []
    if page_match is False:
        issues.append(f"PAGE_MISMATCH: parsed={parsed_count} expected={expected}")
    if missing_pages:
        issues.append(f"MISSING_PAGES: {missing_pages}")
    if empty_pages:
        issues.append(f"EMPTY_PAGES(<50chars): {empty_pages}")
    if duplicate_pages:
        issues.append(f"DUPLICATE_CONSECUTIVE: {duplicate_pages}")
    if coverage is not None and coverage < 0.98:
        issues.append(f"LOW_COVERAGE: {coverage}")

    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "parsed_pages": parsed_count,
        "expected_pages": expected,
        "page_match": page_match,
        "missing_pages": missing_pages,
        "empty_pages": empty_pages,
        "short_pages": short_pages,
        "duplicate_pages": duplicate_pages,
        "coverage_ratio": coverage,
        "avg_chars": avg_chars,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM 파싱 결과 검증")
    parser.add_argument("--parsed-dir", required=True, help="VLM 파싱 결과 디렉토리")
    parser.add_argument("--source-dir", required=True, help="원본 파일 디렉토리")
    parser.add_argument("--output", default=None, help="검증 리포트 JSON 출력 경로")
    parser.add_argument("--fail-on-mismatch", action="store_true", help="페이지 불일치 시 exit 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    source_dir = Path(args.source_dir)

    print(f"Building source file map from {source_dir}...")
    source_map = build_source_map(source_dir)
    print(f"  Found {len(source_map)} source files")

    parsed_files = sorted(parsed_dir.rglob("*.json"))
    # validation_report.json 자체는 제외
    parsed_files = [f for f in parsed_files if f.name != "validation_report.json"]
    print(f"Found {len(parsed_files)} parsed JSON files\n")

    results: list[dict[str, Any]] = []
    summary = {
        "total": 0, "page_match": 0, "page_mismatch": 0, "page_unknown": 0,
        "has_empty_pages": 0, "has_missing_pages": 0, "has_duplicates": 0,
        "low_coverage": 0, "total_parsed_pages": 0, "total_expected_pages": 0,
    }
    issues_by_type: dict[str, int] = defaultdict(int)

    for pf in parsed_files:
        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR loading {pf}: {e}")
            continue

        source_file = data.get("source_file", "")
        source_path = source_map.get(source_file)
        result = validate_single(data, source_path)
        results.append(result)

        summary["total"] += 1
        summary["total_parsed_pages"] += result["parsed_pages"]
        if result["expected_pages"]:
            summary["total_expected_pages"] += result["expected_pages"]

        if result["page_match"] is True:
            summary["page_match"] += 1
        elif result["page_match"] is False:
            summary["page_mismatch"] += 1
        else:
            summary["page_unknown"] += 1

        if result["empty_pages"]:
            summary["has_empty_pages"] += 1
        if result["missing_pages"]:
            summary["has_missing_pages"] += 1
        if result["duplicate_pages"]:
            summary["has_duplicates"] += 1
        if result["coverage_ratio"] is not None and result["coverage_ratio"] < 0.98:
            summary["low_coverage"] += 1

        for issue in result["issues"]:
            issue_type = issue.split(":")[0]
            issues_by_type[issue_type] += 1

    print("=" * 70)
    print("VLM VALIDATION REPORT")
    print("=" * 70)
    print(f"  Documents checked:    {summary['total']}")
    print(f"  Page match:           {summary['page_match']}")
    print(f"  Page mismatch:        {summary['page_mismatch']}")
    print(f"  Page count unknown:   {summary['page_unknown']}")
    print(f"  Total parsed pages:   {summary['total_parsed_pages']}")
    print(f"  Total expected pages: {summary['total_expected_pages']}")
    print()
    print(f"  Has empty pages:      {summary['has_empty_pages']}")
    print(f"  Has missing pages:    {summary['has_missing_pages']}")
    print(f"  Has duplicates:       {summary['has_duplicates']}")
    print(f"  Low coverage (<0.98): {summary['low_coverage']}")
    print()

    if issues_by_type:
        print("Issue breakdown:")
        for itype, count in sorted(issues_by_type.items()):
            print(f"  {itype}: {count}")
        print()

    problem_docs = [r for r in results if r["issues"]]
    if problem_docs:
        print(f"PROBLEM DOCUMENTS ({len(problem_docs)}):")
        print("-" * 70)
        for r in problem_docs:
            print(f"  {r['source_file']}")
            print(f"    parsed={r['parsed_pages']} expected={r['expected_pages']} coverage={r['coverage_ratio']}")
            for issue in r["issues"]:
                print(f"    - {issue}")
            print()
    else:
        print("No issues found!")

    report = {"summary": summary, "issues_by_type": dict(issues_by_type), "documents": results}
    out_path = Path(args.output) if args.output else Path("data/vlm_parsed/validation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to: {out_path}")

    if args.fail_on_mismatch and summary["page_mismatch"] > 0:
        print(f"\nFAILED: {summary['page_mismatch']} documents with page mismatch")
        sys.exit(1)


if __name__ == "__main__":
    main()
