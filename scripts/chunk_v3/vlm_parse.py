"""VLM 파싱 + JSON 저장 + 리뷰 MD 생성 CLI 스크립트.

PDF/PPTX 파일을 VLM으로 파싱하고 결과를 JSON으로 저장한다.
각 문서별로 원본 이미지 + 추출 텍스트를 비교할 수 있는 review.md를 생성한다.
ES 적재는 하지 않음 (Phase 1 전용).

Usage:
    python scripts/chunk_v3/vlm_parse.py \
        --input sops/ --doc-type sop \
        --output data/vlm_parsed/

    # GPU 2개 병렬 처리
    python scripts/chunk_v3/vlm_parse.py \
        --input sops/ --doc-type sop \
        --output data/vlm_parsed/ --workers 2

    # 리뷰 MD 생성 끄기
    python scripts/chunk_v3/vlm_parse.py \
        --input sops/ --doc-type sop \
        --output data/vlm_parsed/ --no-review
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.common import save_vlm_result


def find_parseable_files(input_path: Path) -> list[Path]:
    """입력 경로에서 PDF/PPTX 파일 목록 반환."""
    if input_path.is_file():
        if input_path.suffix.lower() in (".pdf", ".pptx"):
            return [input_path]
        raise ValueError(f"Not a PDF/PPTX file: {input_path}")

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in (".pdf", ".pptx")
        )
        if not files:
            raise FileNotFoundError(f"No PDF/PPTX files found in: {input_path}")
        return files

    raise FileNotFoundError(f"Path not found: {input_path}")


def generate_doc_id(file_path: Path, base_path: Path | None = None) -> str:
    """파일 경로에서 doc_id 생성."""
    if base_path and base_path.is_dir():
        rel_path = file_path.relative_to(base_path)
        name = str(rel_path.with_suffix(""))
    else:
        name = file_path.stem

    doc_id = re.sub(r"[^a-zA-Z0-9가-힣]", "_", name)
    doc_id = re.sub(r"_+", "_", doc_id)
    doc_id = doc_id.strip("_").lower()
    return doc_id or "doc"


def render_page_images(input_path: Path) -> list:
    """PDF/PPTX -> 페이지별 PIL Image 리스트 반환."""
    from pdf2image import convert_from_path, convert_from_bytes

    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        return convert_from_path(str(input_path), dpi=150)

    if suffix == ".pptx":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pptx_copy = tmpdir_path / "input.pptx"
            pptx_copy.write_bytes(input_path.read_bytes())
            try:
                subprocess.run(
                    [
                        "libreoffice", "--headless",
                        "--convert-to", "pdf",
                        "--outdir", str(tmpdir_path),
                        str(pptx_copy),
                    ],
                    capture_output=True, text=True, check=True, timeout=120,
                )
            except Exception:
                return []
            pdf_path = tmpdir_path / "input.pdf"
            if not pdf_path.exists():
                return []
            return convert_from_path(str(pdf_path), dpi=150)

    return []


def generate_review_md(
    doc_id: str,
    doc_type: str,
    source_file: str,
    vlm_model: str,
    pages: list[dict],
    images: list,
    output_dir: str,
) -> Path | None:
    """원본 이미지 + 추출 텍스트를 비교하는 review.md 생성."""
    if not images:
        return None

    review_dir = Path(output_dir) / "review" / doc_type / doc_id
    img_dir = review_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 저장
    for i, img in enumerate(images, start=1):
        img_path = img_dir / f"page_{i:03d}.png"
        img.save(str(img_path), "PNG")

    # review.md 생성
    md_path = review_dir / "review.md"
    lines = [
        f"# {source_file} - VLM 파싱 리뷰",
        f"",
        f"- **doc_id**: `{doc_id}`",
        f"- **doc_type**: `{doc_type}`",
        f"- **모델**: `{vlm_model}`",
        f"- **총 페이지**: {len(pages)}",
        f"",
    ]

    for page_data in pages:
        page_no = page_data["page"]
        text = page_data["text"]
        char_count = len(text)

        lines.append(f"---")
        lines.append(f"## Page {page_no}")
        lines.append(f"")
        lines.append(f"![page {page_no}](images/page_{page_no:03d}.png)")
        lines.append(f"")
        lines.append(f"### 추출 텍스트 ({char_count}자)")
        lines.append(f"")
        lines.append(f"````markdown")
        lines.append(text)
        lines.append(f"````")
        lines.append(f"")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def parse_and_save(
    input_path: Path,
    doc_type: str,
    doc_id: str,
    output_dir: str,
    vlm_model: str | None = None,
    generate_review: bool = True,
) -> dict:
    """단일 PDF/PPTX -> VLM 파싱 -> JSON 저장 + 리뷰 MD 생성."""
    from backend.config.settings import vlm_client_settings
    from backend.llm_infrastructure.vlm.clients import OpenAIVisionClient
    from backend.llm_infrastructure.preprocessing.parsers.base import PdfParseOptions

    vlm_client = OpenAIVisionClient(
        base_url=vlm_client_settings.base_url,
        model=vlm_client_settings.model,
        timeout=vlm_client_settings.timeout,
    )

    model_name = vlm_model or vlm_client_settings.model
    suffix = input_path.suffix.lower()

    if suffix == ".pptx":
        from backend.llm_infrastructure.preprocessing.parsers.engines.pptx_vlm_engine import (
            PptxVlmEngine,
        )
        engine = PptxVlmEngine(vlm_client=vlm_client)
        source_type = "pptx"
    else:
        from backend.llm_infrastructure.preprocessing.parsers.engines.pdf_vlm_engine import (
            VlmPdfEngine,
        )
        engine = VlmPdfEngine(vlm_client=vlm_client)
        source_type = "pdf"

    opts = PdfParseOptions(vlm_model=model_name)
    with open(input_path, "rb") as f:
        parsed = engine.run(f, opts)

    pages = [
        {"page": p.number, "text": p.text, "confidence": None}
        for p in parsed.pages
    ]

    out_path = save_vlm_result(
        doc_type=doc_type,
        doc_id=doc_id,
        pages=pages,
        source_file=input_path.name,
        vlm_model=model_name,
        output_dir=output_dir,
        source_type=source_type,
    )

    # 리뷰 MD 생성
    review_path = None
    if generate_review:
        try:
            images = render_page_images(input_path)
            review_path = generate_review_md(
                doc_id=doc_id,
                doc_type=doc_type,
                source_file=input_path.name,
                vlm_model=model_name,
                pages=pages,
                images=images,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"  review MD 생성 실패 (무시): {e}", flush=True)

    return {
        "doc_id": doc_id,
        "pages": len(pages),
        "output": str(out_path),
        "review": str(review_path) if review_path else None,
    }


def _process_one(
    file_path: Path, doc_type: str, doc_id: str, output_dir: str,
    generate_review: bool = True,
) -> dict:
    """단일 파일 처리 (워커에서 호출)."""
    start = time.time()
    result = parse_and_save(
        input_path=file_path,
        doc_type=doc_type,
        doc_id=doc_id,
        output_dir=output_dir,
        generate_review=generate_review,
    )
    elapsed = time.time() - start
    result["elapsed"] = round(elapsed, 1)
    result["file_name"] = file_path.name
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM PDF/PPTX parsing -> JSON storage")
    parser.add_argument(
        "--input", required=True,
        help="PDF/PPTX 파일 또는 폴더 경로",
    )
    parser.add_argument(
        "--doc-type", required=True,
        help="문서 종류 (sop, sop_pptx, ts, setup_manual)",
    )
    parser.add_argument(
        "--output", default="data/vlm_parsed",
        help="출력 디렉토리 (기본: data/vlm_parsed)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="동시 처리 워커 수 (기본: 1, GPU 수에 맞춤)",
    )
    parser.add_argument(
        "--no-review", action="store_true",
        help="리뷰 MD 생성 안 함 (이미지+텍스트 비교 파일)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    files = find_parseable_files(input_path)
    base_path = input_path if input_path.is_dir() else None
    workers = max(1, args.workers)

    print(f"Found {len(files)} file(s) to process", flush=True)
    print(f"  doc_type: {args.doc_type}", flush=True)
    print(f"  output: {args.output}", flush=True)
    generate_review = not args.no_review
    print(f"  workers: {workers}", flush=True)
    print(f"  review MD: {'ON' if generate_review else 'OFF'}", flush=True)

    # 이미 파싱된 파일 스킵 (resume 지원)
    out_dir = Path(args.output) / args.doc_type
    todo: list[tuple[Path, str]] = []
    skipped = 0
    for file_path in files:
        doc_id = generate_doc_id(file_path, base_path)
        existing = out_dir / f"{doc_id}.json"
        if existing.exists():
            skipped += 1
            continue
        todo.append((file_path, doc_id))

    if skipped:
        print(f"  skipped (already parsed): {skipped}", flush=True)
    print(f"  remaining: {len(todo)}", flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
        return

    success = 0
    failed: list[tuple[str, str]] = []
    total_start = time.time()

    if workers == 1:
        for i, (file_path, doc_id) in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {file_path.name} -> {doc_id}", flush=True)
            try:
                result = _process_one(file_path, args.doc_type, doc_id, args.output, generate_review)
                review_msg = f", review: {result['review']}" if result.get('review') else ""
                print(f"  OK: {result['pages']}p, {result['elapsed']}s{review_msg}", flush=True)
                success += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                failed.append((file_path.name, str(e)))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for file_path, doc_id in todo:
                fut = executor.submit(
                    _process_one, file_path, args.doc_type, doc_id, args.output, generate_review,
                )
                futures[fut] = (file_path.name, doc_id)

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                fname, doc_id = futures[fut]
                try:
                    result = fut.result()
                    print(
                        f"  [{done_count}/{len(todo)}] {fname}: "
                        f"{result['pages']}p, {result['elapsed']}s",
                        flush=True,
                    )
                    success += 1
                except Exception as e:
                    print(f"  [{done_count}/{len(todo)}] {fname}: ERROR {e}", flush=True)
                    failed.append((fname, str(e)))

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"Total: {len(todo)}, Success: {success}, Failed: {len(failed)}", flush=True)
    print(f"Elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)
    if failed:
        print("Failed files:", flush=True)
        for fname, err in failed:
            print(f"  - {fname}: {err}", flush=True)


if __name__ == "__main__":
    main()
