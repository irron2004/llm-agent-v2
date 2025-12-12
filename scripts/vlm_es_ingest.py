"""Run VLM-based PDF parsing and ingest results into Elasticsearch.

Usage (example):
    python scripts/vlm_es_ingest.py \
        --pdf "data/global sop_supra xp_all_efem_rfid assy.pdf" \
        --doc-id global_sop_efem_rfid \
        --doc-type sop \
        --lang ko \
        --tenant-id tenant1 \
        --project-id proj1 \
        --tags manual equipment \
        --refresh

    # 페이지 이미지 저장 및 ES에 경로 포함
    python scripts/vlm_es_ingest.py \
        --pdf "data/sample.pdf" \
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM PDF ingest to Elasticsearch")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--doc-id", required=True, help="Base document id for ES")
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


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) VLM client (OpenAI-compatible vision API)
    vlm_client = OpenAIVisionClient(
        base_url=vlm_client_settings.base_url,
        model=vlm_client_settings.model,
        timeout=vlm_client_settings.timeout,
    )

    # 2) VLM parsing
    ingest_svc = DocumentIngestService.for_vlm(vlm_client=vlm_client)
    with pdf_path.open("rb") as f:
        parsed = ingest_svc.ingest_pdf(f, doc_type=args.doc_type)

    sections = parsed.get("sections", [])
    if args.max_sections:
        sections = sections[: args.max_sections]

    # 3) 페이지 이미지 저장 (optional)
    page_image_paths: Optional[dict[int, str]] = None
    if args.save_images:
        image_dir = args.image_dir or Path(f"data/page_images/{args.doc_id}")
        page_image_paths = save_page_images(
            pdf_path=pdf_path,
            image_dir=image_dir,
            dpi=args.dpi,
            image_base_url=args.image_base_url,
        )

    # 4) Section 객체로 변환 (이미지 경로 주입 포함)
    section_objs = to_section_objs(sections, page_image_paths)
    print(f"Parsed sections: {len(section_objs)}")
    for i, sec in enumerate(section_objs[:3], 1):
        print(f"[Section {i}] {sec.title}\n{sec.text[:200]}...\n")

    # 5) ES ingest
    alias = args.index or f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"
    es_ingest = EsIngestService.from_settings(index=alias)
    result = es_ingest.ingest_sections(
        base_doc_id=args.doc_id,
        sections=section_objs,
        doc_type=args.doc_type,
        lang=args.lang,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
        vlm_model=vlm_client_settings.model,
        tags=args.tags,
        refresh=args.refresh,
    )

    print(f"Ingested {result['indexed']} chunks into {result['index']}")
    print(f"  doc_type={args.doc_type}, lang={args.lang}, vlm_model={vlm_client_settings.model}")
    if page_image_paths:
        print(f"  page_image_path included for {len(page_image_paths)} pages")


if __name__ == "__main__":
    main()
