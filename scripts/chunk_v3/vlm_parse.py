"""VLM 파싱 + JSON 저장 CLI 스크립트.

PDF/PPTX 파일을 VLM으로 파싱하고 결과를 JSON으로 저장한다.
ES 적재는 하지 않음 (Phase 1 전용).

Usage:
    python scripts/chunk_v3/vlm_parse.py \
        --input sop_pdfs/ --doc-type sop_pdf \
        --output data/vlm_parsed/

    # GPU 2개 병렬 처리
    python scripts/chunk_v3/vlm_parse.py \
        --input sop_pdfs/ --doc-type sop_pdf \
        --output data/vlm_parsed/ --workers 2
"""

from __future__ import annotations

import argparse
import re
import sys
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


def parse_and_save(
    input_path: Path,
    doc_type: str,
    doc_id: str,
    output_dir: str,
    vlm_model: str | None = None,
) -> dict:
    """단일 PDF/PPTX -> VLM 파싱 -> JSON 저장."""
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

    return {
        "doc_id": doc_id,
        "pages": len(pages),
        "output": str(out_path),
    }


def _process_one(file_path: Path, doc_type: str, doc_id: str, output_dir: str) -> dict:
    """단일 파일 처리 (워커에서 호출)."""
    start = time.time()
    result = parse_and_save(
        input_path=file_path,
        doc_type=doc_type,
        doc_id=doc_id,
        output_dir=output_dir,
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
        help="문서 종류 (sop_pdf, sop_pptx, ts, setup_manual)",
    )
    parser.add_argument(
        "--output", default="data/vlm_parsed",
        help="출력 디렉토리 (기본: data/vlm_parsed)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="동시 처리 워커 수 (기본: 1, GPU 수에 맞춤)",
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
    print(f"  workers: {workers}", flush=True)

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
                result = _process_one(file_path, args.doc_type, doc_id, args.output)
                print(f"  OK: {result['pages']}p, {result['elapsed']}s", flush=True)
                success += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                failed.append((file_path.name, str(e)))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for file_path, doc_id in todo:
                fut = executor.submit(
                    _process_one, file_path, args.doc_type, doc_id, args.output,
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
