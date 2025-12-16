"""Run VLM-based PDF parsing and ingest results into Elasticsearch.

Usage (example):
    # 단일 PDF 파일 처리
    python scripts/vlm_es_ingest.py \
        --input "data/global sop_supra xp_all_efem_rfid assy.pdf" \
        --doc-id global_sop_efem_rfid \
        --doc-type sop \
        --lang ko \
        --tenant-id tenant1 \
        --project-id proj1 \
        --tags manual equipment \
        --refresh

    # 폴더 전체 처리 (하위 폴더 포함)
    python scripts/vlm_es_ingest.py \
        --input "data/manuals/" \
        --doc-type sop \
        --lang ko \
        --tenant-id tenant1

    # 페이지 이미지 저장 및 ES에 경로 포함
    python scripts/vlm_es_ingest.py \
        --input "data/sample.pdf" \
        --doc-id sample_001 \
        --save-images \
        --image-dir data/page_images/sample_001

Prerequisites:
- vLLM (OpenAI-compatible) server running with a vision model (e.g., Qwen3-VL)
- Elasticsearch with rag_chunks_* index/alias created
- .env configured (VLM_CLIENT_*, SEARCH_*, RAG_* settings)
- pdf2image + poppler (optional, for --save-images)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure project root is on PYTHONPATH so `backend` imports work when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import search_settings, vlm_client_settings
from backend.services.ingest.document_ingest_service import DocumentIngestService, Section
from backend.llm_infrastructure.vlm.clients import OpenAIVisionClient
from backend.services.es_ingest_service import EsIngestService


def find_pdf_files(input_path: Path) -> List[Path]:
    """입력 경로에서 PDF 파일 목록을 반환.

    Args:
        input_path: PDF 파일 또는 폴더 경로

    Returns:
        PDF 파일 경로 리스트 (정렬됨)
    """
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            return [input_path]
        raise ValueError(f"Not a PDF file: {input_path}")

    if input_path.is_dir():
        pdf_files = sorted(input_path.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {input_path}")
        return pdf_files

    raise FileNotFoundError(f"Path not found: {input_path}")


def generate_doc_id(pdf_path: Path, base_path: Optional[Path] = None) -> str:
    """PDF 경로에서 doc_id 생성.

    Args:
        pdf_path: PDF 파일 경로
        base_path: 기준 폴더 경로 (있으면 상대 경로 기반으로 생성)

    Returns:
        ES용 document id (영문/숫자/언더스코어만 포함)
    """
    if base_path and base_path.is_dir():
        # 폴더 기준 상대 경로 사용 (확장자 제외)
        rel_path = pdf_path.relative_to(base_path)
        name = str(rel_path.with_suffix(""))
    else:
        # 파일명만 사용 (확장자 제외)
        name = pdf_path.stem

    # 특수문자를 언더스코어로 치환, 연속 언더스코어 정리
    doc_id = re.sub(r"[^a-zA-Z0-9가-힣]", "_", name)
    doc_id = re.sub(r"_+", "_", doc_id)
    doc_id = doc_id.strip("_").lower()

    return doc_id or "doc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM PDF ingest to Elasticsearch")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to PDF file or folder (recursively processes all PDFs in folder)",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Base document id for ES (auto-generated from filename if not provided)",
    )
    parser.add_argument("--doc-type", default="generic", help="Document type (e.g., sop, guide)")
    parser.add_argument(
        "--index",
        default=None,
        help="Target ES index/alias (default: rag_chunks_{env}_current)",
    )
    parser.add_argument("--lang", default="ko", help="Language code (default: ko)")
    parser.add_argument("--tenant-id", default="", help="Tenant ID for multi-tenancy")
    parser.add_argument("--project-id", default="", help="Project ID")
    parser.add_argument("--tags", nargs="*", default=None, help="Tags to attach to chunks")
    parser.add_argument("--refresh", action="store_true", help="Refresh index after ingest")
    parser.add_argument("--max-sections", type=int, default=None, help="Limit number of sections (debug)")
    # 페이지 이미지 저장 옵션
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="PDF 페이지를 이미지로 저장하고 ES 문서에 경로 포함",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="이미지 저장 디렉토리 (기본: data/page_images/<doc-id>)",
    )
    parser.add_argument(
        "--image-base-url",
        default=None,
        help="ES에 저장할 이미지 경로 prefix (예: /static/images)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF→이미지 변환 DPI (기본: 150)",
    )
    return parser.parse_args()


def to_section_objs(
    sections: Iterable[dict],
    page_image_paths: Optional[dict[int, str]] = None,
) -> List[Section]:
    """섹션 딕셔너리를 Section 객체로 변환.

    Args:
        sections: 섹션 딕셔너리 리스트
        page_image_paths: 페이지 번호 → 이미지 경로 매핑 (optional)
    """
    objs: List[Section] = []
    for sec in sections:
        metadata = dict(sec.get("metadata", {}))

        # 페이지 이미지 경로 주입
        if page_image_paths:
            page_start = sec.get("page_start")
            if page_start and page_start in page_image_paths:
                metadata["page_image_path"] = page_image_paths[page_start]

        objs.append(
            Section(
                title=sec.get("title", ""),
                text=sec.get("text", ""),
                page_start=sec.get("page_start"),
                page_end=sec.get("page_end"),
                metadata=metadata,
            )
        )
    return objs


def save_page_images(
    pdf_path: Path,
    image_dir: Path,
    dpi: int = 150,
    image_base_url: Optional[str] = None,
) -> dict[int, str]:
    """PDF 페이지를 이미지로 저장하고 페이지 번호 → 경로 매핑 반환.

    Args:
        pdf_path: PDF 파일 경로
        image_dir: 이미지 저장 디렉토리
        dpi: 이미지 해상도
        image_base_url: ES에 저장할 경로 prefix (None이면 로컬 경로 사용)

    Returns:
        페이지 번호(1-based) → 이미지 경로 딕셔너리
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image가 필요합니다. pip install pdf2image && apt install poppler-utils"
        )

    image_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path), dpi=dpi)
    page_paths: dict[int, str] = {}

    for i, img in enumerate(images, start=1):
        filename = f"page_{i:03d}.png"
        local_path = image_dir / filename
        img.save(local_path)

        # ES에 저장할 경로 결정
        if image_base_url:
            page_paths[i] = f"{image_base_url.rstrip('/')}/{filename}"
        else:
            page_paths[i] = str(local_path)

    print(f"Saved {len(images)} page images to {image_dir}")
    return page_paths


def process_single_pdf(
    pdf_path: Path,
    doc_id: str,
    args: argparse.Namespace,
    ingest_svc: DocumentIngestService,
    es_ingest: EsIngestService,
) -> dict:
    """단일 PDF 파일을 처리하고 ES에 색인.

    Args:
        pdf_path: PDF 파일 경로
        doc_id: ES document ID
        args: CLI 인자
        ingest_svc: VLM 기반 문서 파싱 서비스
        es_ingest: ES 색인 서비스

    Returns:
        색인 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"  doc_id: {doc_id}")

    # VLM parsing
    with pdf_path.open("rb") as f:
        parsed = ingest_svc.ingest_pdf(f, doc_type=args.doc_type)

    sections = parsed.get("sections", [])
    if args.max_sections:
        sections = sections[: args.max_sections]

    # 페이지 이미지 저장 (optional)
    page_image_paths: Optional[dict[int, str]] = None
    if args.save_images:
        image_dir = args.image_dir or Path(f"data/page_images/{doc_id}")
        page_image_paths = save_page_images(
            pdf_path=pdf_path,
            image_dir=image_dir,
            dpi=args.dpi,
            image_base_url=args.image_base_url,
        )

    # Section 객체로 변환
    section_objs = to_section_objs(sections, page_image_paths)
    print(f"  Parsed sections: {len(section_objs)}")

    # ES ingest
    result = es_ingest.ingest_sections(
        base_doc_id=doc_id,
        sections=section_objs,
        doc_type=args.doc_type,
        lang=args.lang,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
        vlm_model=vlm_client_settings.model,
        tags=args.tags,
        refresh=args.refresh,
    )

    print(f"  Ingested {result['indexed']} chunks")
    if page_image_paths:
        print(f"  Page images: {len(page_image_paths)}")

    return result


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)

    # PDF 파일 목록 찾기
    pdf_files = find_pdf_files(input_path)
    print(f"Found {len(pdf_files)} PDF file(s) to process")

    # 기준 경로 설정 (폴더인 경우 해당 폴더, 아니면 None)
    base_path = input_path if input_path.is_dir() else None

    # VLM client 초기화 (재사용)
    vlm_client = OpenAIVisionClient(
        base_url=vlm_client_settings.base_url,
        model=vlm_client_settings.model,
        timeout=vlm_client_settings.timeout,
    )
    ingest_svc = DocumentIngestService.for_vlm(vlm_client=vlm_client)

    # ES ingest 서비스 초기화
    alias = args.index or f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"
    es_ingest = EsIngestService.from_settings(index=alias)

    # 각 PDF 처리
    total_indexed = 0
    success_count = 0
    failed_files: List[Path] = []

    for pdf_path in pdf_files:
        # doc_id 결정: CLI에서 제공된 경우 사용, 아니면 자동 생성
        if args.doc_id and len(pdf_files) == 1:
            doc_id = args.doc_id
        else:
            doc_id = generate_doc_id(pdf_path, base_path)

        try:
            result = process_single_pdf(
                pdf_path=pdf_path,
                doc_id=doc_id,
                args=args,
                ingest_svc=ingest_svc,
                es_ingest=es_ingest,
            )
            total_indexed += result.get("indexed", 0)
            success_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_files.append(pdf_path)

    # 최종 요약
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Total files: {len(pdf_files)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(failed_files)}")
    print(f"  Total chunks indexed: {total_indexed}")
    print(f"  Index: {alias}")
    print(f"  doc_type={args.doc_type}, lang={args.lang}, vlm_model={vlm_client_settings.model}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
