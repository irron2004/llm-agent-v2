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

    # 여러 문서 병렬 처리 (워커 수 직접 지정)
    python scripts/vlm_es_ingest.py \
        --input "data/manuals/" \
        --doc-type sop \
        --lang ko \
        --tenant-id tenant1 \
        --workers 2

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
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure project root is on PYTHONPATH so `backend` imports work when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import search_settings, vlm_client_settings, ingest_settings
from backend.services.ingest.document_ingest_service import DocumentIngestService, Section
from backend.services.ingest.metadata_extractor import create_metadata_extractor
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
    # 메타데이터 추출 옵션
    parser.add_argument(
        "--enable-metadata",
        action="store_true",
        default=True,
        help="문서 메타데이터 추출 활성화 (device_name, doc_description)",
    )
    parser.add_argument(
        "--disable-metadata",
        action="store_true",
        help="문서 메타데이터 추출 비활성화",
    )
    parser.add_argument(
        "--enable-chapters",
        action="store_true",
        default=True,
        help="챕터 제목 추출 활성화 (carry-forward)",
    )
    parser.add_argument(
        "--disable-chapters",
        action="store_true",
        help="챕터 제목 추출 비활성화",
    )
    parser.add_argument(
        "--enable-summaries",
        action="store_true",
        help="청크별 요약 생성 활성화 (LLM 필요, 느림)",
    )
    parser.add_argument(
        "--use-llm-fallback",
        action="store_true",
        default=True,
        help="규칙 기반 추출 실패 시 LLM 사용",
    )
    parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="LLM fallback 비활성화 (규칙 기반만 사용)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="동시 처리할 PDF 수 (기본: GPU 상태 기반 자동 계산)",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="자동 워커 계산 시 최소 워커 수",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="자동 워커 계산 시 최대 워커 수",
    )
    parser.add_argument(
        "--gpu-mem-per-request-gb",
        type=float,
        default=4.0,
        help="동시 VLM 요청당 예상 GPU 메모리 (GB)",
    )
    parser.add_argument(
        "--gpu-safety-margin-gb",
        type=float,
        default=4.0,
        help="GPU 메모리 안전 여유분 (GB)",
    )
    parser.add_argument(
        "--auto-workers",
        dest="auto_workers",
        action="store_true",
        help="GPU 상태 기반 자동 워커 계산 활성화 (기본값)",
    )
    parser.add_argument(
        "--no-auto-workers",
        dest="auto_workers",
        action="store_false",
        help="GPU 상태 기반 자동 워커 계산 비활성화",
    )
    parser.set_defaults(auto_workers=True)
    return parser.parse_args()


def _parse_visible_gpu_indices(raw: Optional[str], total: int) -> Optional[List[int]]:
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() == "all":
        return None
    indices: List[int] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < total:
                indices.append(idx)
    return indices or None


def _get_gpu_free_gb() -> Optional[List[float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    free_mib: List[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            free_mib.append(float(line))
        except ValueError:
            continue

    if not free_mib:
        return None

    free_gib = [mib / 1024.0 for mib in free_mib]
    visible_raw = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("VLM_CUDA_DEVICES")
        or os.environ.get("VLLM_CUDA_DEVICES")
    )
    indices = _parse_visible_gpu_indices(visible_raw, len(free_gib))
    if indices:
        free_gib = [free_gib[i] for i in indices]

    return free_gib or None


def _resolve_workers(args: argparse.Namespace, pdf_count: int) -> int:
    if args.workers is not None:
        return max(1, min(args.workers, pdf_count))

    if not args.auto_workers:
        return 1

    gpu_free_gb = _get_gpu_free_gb()
    if not gpu_free_gb:
        print("Auto workers: nvidia-smi unavailable; defaulting to 1")
        return 1

    min_free_gb = min(gpu_free_gb)
    safety = max(args.gpu_safety_margin_gb, 0.0)
    per_req = max(args.gpu_mem_per_request_gb, 0.1)
    available = max(min_free_gb - safety, 0.0)
    workers = int(available // per_req)
    workers = max(args.min_workers, workers)
    if args.max_workers is not None:
        workers = min(workers, args.max_workers)
    workers = max(1, min(workers, pdf_count))

    print(
        "Auto workers: min_free_gb={:.1f}, safety_gb={:.1f}, per_req_gb={:.1f} -> {}".format(
            min_free_gb, safety, per_req, workers
        )
    )
    return workers


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

    # 메타데이터 추출 설정 결정
    enable_doc_metadata = not args.disable_metadata
    enable_chapters = not args.disable_chapters
    enable_summaries = args.enable_summaries
    use_llm_fallback = not args.no_llm_fallback

    # MetadataExtractor 사용 여부 (실제 생성은 워커 단위로 수행)
    needs_metadata_extractor = enable_doc_metadata or enable_chapters or enable_summaries
    if needs_metadata_extractor:
        print(
            "Metadata extraction: doc_meta={}, chapters={}, summaries={}".format(
                enable_doc_metadata, enable_chapters, enable_summaries
            )
        )

    # ES ingest 대상 index alias
    alias = args.index or f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

    # Thread-local services to avoid shared clients across workers.
    thread_state = threading.local()

    def get_services() -> tuple[DocumentIngestService, EsIngestService]:
        ingest_svc = getattr(thread_state, "ingest_svc", None)
        es_ingest = getattr(thread_state, "es_ingest", None)

        if ingest_svc is None:
            local_metadata_extractor = None
            if needs_metadata_extractor:
                local_metadata_extractor = create_metadata_extractor(
                    use_llm_fallback=use_llm_fallback
                )

            vlm_client = OpenAIVisionClient(
                base_url=vlm_client_settings.base_url,
                model=vlm_client_settings.model,
                timeout=vlm_client_settings.timeout,
            )
            ingest_svc = DocumentIngestService.for_vlm(
                vlm_client=vlm_client,
                metadata_extractor=local_metadata_extractor,
                enable_doc_metadata=enable_doc_metadata,
                enable_chapter_extraction=enable_chapters,
                enable_summarization=enable_summaries,
            )
            thread_state.ingest_svc = ingest_svc

        if es_ingest is None:
            es_ingest = EsIngestService.from_settings(index=alias)
            thread_state.es_ingest = es_ingest

        return ingest_svc, es_ingest

    # 각 PDF 처리
    total_indexed = 0
    success_count = 0
    failed_files: List[Path] = []

    tasks: List[tuple[Path, str]] = []
    for pdf_path in pdf_files:
        if args.doc_id and len(pdf_files) == 1:
            doc_id = args.doc_id
        else:
            doc_id = generate_doc_id(pdf_path, base_path)
        tasks.append((pdf_path, doc_id))

    workers = _resolve_workers(args, len(tasks))
    if workers <= 1:
        for pdf_path, doc_id in tasks:
            try:
                ingest_svc, es_ingest = get_services()
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
    else:
        print(f"Running with {workers} worker(s)")

        def _worker(pdf_path: Path, doc_id: str) -> dict:
            ingest_svc, es_ingest = get_services()
            return process_single_pdf(
                pdf_path=pdf_path,
                doc_id=doc_id,
                args=args,
                ingest_svc=ingest_svc,
                es_ingest=es_ingest,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_worker, pdf_path, doc_id): (pdf_path, doc_id)
                for pdf_path, doc_id in tasks
            }
            for future in as_completed(future_map):
                pdf_path, _doc_id = future_map[future]
                try:
                    result = future.result()
                    total_indexed += result.get("indexed", 0)
                    success_count += 1
                except Exception as e:
                    print(f"  ERROR: {pdf_path} -> {e}")
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
    print(f"  Metadata: doc_meta={enable_doc_metadata}, chapters={enable_chapters}, summaries={enable_summaries}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
