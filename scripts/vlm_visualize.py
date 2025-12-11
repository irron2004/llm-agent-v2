"""VLM 파싱 결과를 폴더별로 시각화하는 배치 스크립트.

문서별로 원본 PDF, 페이지 이미지, VLM 파싱 결과를 폴더 구조로 저장하고
검증용 HTML 프리뷰를 생성합니다.

Usage:
    # 단일 PDF 시각화
    python scripts/vlm_visualize.py --pdf data/sample.pdf

    # 폴더 내 모든 PDF 배치 시각화
    python scripts/vlm_visualize.py --input-dir data/vlm_eval

    # 커스텀 출력 경로
    python scripts/vlm_visualize.py --input-dir data/vlm_eval --output-dir experiments/vlm_runs

출력 구조:
    experiments/vlm_previews/<timestamp>/
    ├── index.html                    # 전체 문서 목록
    └── <doc_name>/
        ├── source.pdf                # 원본 PDF 복사본
        ├── pages/
        │   ├── page_001.png          # 페이지 이미지
        │   └── ...
        ├── vlm/
        │   ├── page_001.txt          # 페이지별 VLM 텍스트
        │   └── ...
        ├── sections.json             # 섹션 분리 결과
        └── preview.html              # 시각화 HTML (이미지 ↔ 텍스트)

Prerequisites:
    - pdf2image: pip install pdf2image
    - poppler: apt install poppler-utils (Linux) / brew install poppler (Mac)
    - VLM 서버 실행 중 (.env의 VLM_CLIENT_* 설정)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

# 프로젝트 루트를 sys.path에 추가 (IDE에서 직접 실행 시 필요)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pdf2image import convert_from_bytes

from backend.config.settings import vlm_client_settings, vlm_parser_settings
from backend.llm_infrastructure.vlm.clients import OpenAIVisionClient
from backend.services.ingest.document_ingest_service import DocumentIngestService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM 파싱 결과 시각화 및 검증용 배치 스크립트"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        # default=Path("data/global sop_supra xp_all_efem_rfid assy.pdf"),
        help="단일 PDF 파일 경로",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/sop_pdfs"),
        help="PDF들이 있는 입력 디렉토리 (하위 폴더 포함 재귀 탐색)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ingestions"),
        help="출력 기본 디렉토리 (기본: experiments/vlm_previews)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF→이미지 변환 DPI (기본: 150)",
    )
    parser.add_argument(
        "--extract-figures",
        action="store_true",
        help="VLM 텍스트의 <!-- Image (x0, y0, x1, y1) --> 마커를 잘라 이미지로 저장",
    )
    parser.add_argument(
        "--bbox-origin",
        choices=["top-left", "bottom-left"],
        default="top-left",
        help="좌표계 기준 (기본: top-left). bottom-left는 PDF 좌표계에 맞출 때 사용",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="타임스탬프 폴더 없이 바로 output-dir에 저장",
    )
    parser.add_argument(
        "--doc-type",
        default="generic",
        help="문서 타입 (sop, guide, etc.) - 파싱 힌트로 사용",
    )
    return parser.parse_args()


def img_to_base64(img) -> str:
    """PIL 이미지를 base64 문자열로 변환."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


MARKER_RE = re.compile(r"<!--\s*Image\s*\(([^)]+)\)\s*-->")


def _parse_bboxes(text: str) -> List[Tuple[Tuple[int, int], Sequence[float]]]:
    """추출된 텍스트에서 이미지 마커와 좌표를 파싱.

    Returns:
        [(span, (x0, y0, x1, y1)), ...]
    """
    matches: List[Tuple[Tuple[int, int], Sequence[float]]] = []
    for m in MARKER_RE.finditer(text or ""):
        coords_raw = m.group(1)
        try:
            coords = [float(p.strip()) for p in coords_raw.split(",")]
            if len(coords) != 4:
                continue
        except Exception:
            continue
        matches.append(((m.start(), m.end()), coords))
    return matches


def _scale_bbox(
    coords: Sequence[float],
    img_width: int,
    img_height: int,
    origin: str,
) -> Tuple[int, int, int, int]:
    """좌표를 이미지 픽셀 좌표로 변환 (클램프 포함)."""
    x0, y0, x1, y1 = coords
    base_w = max(x0, x1, img_width)
    base_h = max(y0, y1, img_height)
    scale_x = img_width / base_w
    scale_y = img_height / base_h

    if origin == "bottom-left":
        top = img_height - (y1 * scale_y)
        bottom = img_height - (y0 * scale_y)
    else:
        top = y0 * scale_y
        bottom = y1 * scale_y

    left = x0 * scale_x
    right = x1 * scale_x

    left_i = int(max(0, min(img_width, left)))
    right_i = int(max(0, min(img_width, right)))
    top_i = int(max(0, min(img_height, top)))
    bottom_i = int(max(0, min(img_height, bottom)))

    if right_i <= left_i:
        right_i = min(img_width, left_i + 1)
    if bottom_i <= top_i:
        bottom_i = min(img_height, top_i + 1)

    return left_i, top_i, right_i, bottom_i


def _replace_with_image_links(
    text: str,
    spans_and_paths: List[Tuple[Tuple[int, int], str]],
) -> str:
    """마커를 이미지 링크로 교체."""
    if not spans_and_paths:
        return text
    # 뒤에서부터 치환해야 위치 어긋남이 없다
    spans_and_paths.sort(key=lambda x: x[0][0], reverse=True)
    new_text = text
    for (start, end), rel_path in spans_and_paths:
        replacement = f"![extracted image]({rel_path})"
        new_text = new_text[:start] + replacement + new_text[end:]
    return new_text


def generate_preview_html(
    doc_name: str,
    pages: list,
    images: list,
    sections: list,
    vlm_model: str,
    vlm_prompt: str,
) -> str:
    """페이지별 이미지 ↔ 텍스트 비교 HTML 생성."""
    sections_html = []
    for i, (page, img) in enumerate(zip(pages, images), start=1):
        b64 = img_to_base64(img)
        text = html.escape(page.text or "")
        sections_html.append(f"""
        <section class="page-section">
            <h3>Page {i}</h3>
            <div class="page-content">
                <div class="page-image">
                    <img src="data:image/png;base64,{b64}" alt="Page {i}">
                </div>
                <div class="page-text">
                    <pre>{text}</pre>
                </div>
            </div>
        </section>
        """)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VLM Preview - {html.escape(doc_name)}</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 24px;
            background: #fafafa;
        }}
        h1 {{ color: #333; }}
        .meta {{
            background: #e8f4f8;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .meta code {{
            background: #fff;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .page-section {{
            margin-bottom: 32px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
        }}
        .page-section h3 {{
            margin-top: 0;
            color: #555;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        .page-content {{
            display: flex;
            gap: 16px;
        }}
        .page-image {{
            flex: 1;
            min-width: 0;
        }}
        .page-image img {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .page-text {{
            flex: 1;
            min-width: 0;
        }}
        .page-text pre {{
            margin: 0;
            padding: 12px;
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 13px;
            line-height: 1.5;
            max-height: 800px;
            overflow-y: auto;
        }}
        .sections-summary {{
            margin-top: 32px;
            padding: 16px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .sections-summary h2 {{
            margin-top: 0;
        }}
        .section-item {{
            margin: 12px 0;
            padding: 12px;
            background: #f5f5f5;
            border-radius: 4px;
        }}
        .section-item .title {{
            font-weight: bold;
            color: #333;
        }}
        .section-item .pages {{
            font-size: 12px;
            color: #666;
        }}
        .section-item .preview {{
            margin-top: 8px;
            font-size: 13px;
            color: #555;
        }}
    </style>
</head>
<body>
    <h1>VLM Preview: {html.escape(doc_name)}</h1>
    <div class="meta">
        <strong>Model:</strong> <code>{html.escape(vlm_model)}</code><br>
        <strong>Prompt:</strong> <code>{html.escape(vlm_prompt[:200] + '...' if len(vlm_prompt) > 200 else vlm_prompt)}</code><br>
        <strong>Total Pages:</strong> {len(pages)}<br>
        <strong>Total Sections:</strong> {len(sections)}
    </div>

    <h2>Page-by-Page Comparison</h2>
    {''.join(sections_html)}

    <div class="sections-summary">
        <h2>Extracted Sections</h2>
        {''.join(f'''
        <div class="section-item">
            <div class="title">{html.escape(sec.get("title", "(untitled)"))}</div>
            <div class="pages">Pages {sec.get("page_start", "?")} - {sec.get("page_end", "?")}</div>
            <div class="preview">{html.escape(sec.get("text", "")[:300])}...</div>
        </div>
        ''' for sec in sections)}
    </div>
</body>
</html>
"""


def generate_index_html(run_dir: Path, entries: List[dict]) -> str:
    """전체 문서 목록 인덱스 HTML 생성."""
    items = []
    for e in entries:
        items.append(f"""
        <li>
            <a href="{e['preview_path']}">{html.escape(e['name'])}</a>
            <span class="meta">({e['pages']} pages, {e['sections']} sections)</span>
        </li>
        """)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VLM Batch Preview</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 24px;
            background: #fafafa;
        }}
        h1 {{ color: #333; }}
        .info {{
            background: #e8f4f8;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 24px;
        }}
        ul {{
            list-style: none;
            padding: 0;
        }}
        li {{
            padding: 12px;
            margin: 8px 0;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 6px;
        }}
        li a {{
            font-weight: bold;
            color: #0066cc;
            text-decoration: none;
        }}
        li a:hover {{
            text-decoration: underline;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-left: 12px;
        }}
    </style>
</head>
<body>
    <h1>VLM Batch Preview</h1>
    <div class="info">
        <strong>Run Directory:</strong> {run_dir}<br>
        <strong>Total Documents:</strong> {len(entries)}<br>
        <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    <ul>
        {''.join(items)}
    </ul>
</body>
</html>
"""


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    svc: DocumentIngestService,
    dpi: int,
    doc_type: str,
    extract_figures: bool,
    bbox_origin: str,
) -> dict:
    """단일 PDF 처리: 파싱 → 이미지 변환 → 저장."""
    print(f"Processing: {pdf_path}")

    pages_dir = out_dir / "pages"
    vlm_dir = out_dir / "vlm"
    pages_dir.mkdir(parents=True, exist_ok=True)
    vlm_dir.mkdir(parents=True, exist_ok=True)

    # 원본 PDF 복사
    shutil.copy2(pdf_path, out_dir / "source.pdf")

    # PDF 바이트 읽기
    pdf_bytes = pdf_path.read_bytes()

    # VLM 파싱
    with pdf_path.open("rb") as f:
        result = svc.ingest_pdf(f, doc_type=doc_type)

    pages = result["parsed"].pages
    sections = result.get("sections", [])

    # PDF → 이미지 변환
    images = convert_from_bytes(pdf_bytes, dpi=dpi)

    # 페이지 내부 이미지 마커 추출 및 크롭 저장
    if extract_figures:
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for i, (page, img) in enumerate(zip(pages, images), start=1):
            markers = _parse_bboxes(page.text or "")
            spans_and_paths: List[tuple[tuple[int, int], str]] = []
            for j, (span, coords) in enumerate(markers, start=1):
                left, top, right, bottom = _scale_bbox(
                    coords,
                    img_width=img.width,
                    img_height=img.height,
                    origin=bbox_origin,
                )
                cropped = img.crop((left, top, right, bottom))
                fig_name = f"page_{i:03d}_fig_{j:02d}.png"
                fig_path = figures_dir / fig_name
                cropped.save(fig_path)
                spans_and_paths.append((span, f"figures/{fig_name}"))

            # 마커를 이미지 링크로 교체해 텍스트에 반영 (optional)
            if spans_and_paths:
                page.text = _replace_with_image_links(page.text or "", spans_and_paths)

    # 섹션 JSON 저장
    sections_path = out_dir / "sections.json"
    sections_path.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 페이지별 이미지 및 텍스트 저장 (마커 치환 후 텍스트 반영)
    for i, (page, img) in enumerate(zip(pages, images), start=1):
        img_path = pages_dir / f"page_{i:03d}.png"
        txt_path = vlm_dir / f"page_{i:03d}.txt"

        img.save(img_path)
        txt_path.write_text(page.text or "", encoding="utf-8")

    # 프리뷰 HTML 생성
    preview_html = generate_preview_html(
        doc_name=pdf_path.name,
        pages=pages,
        images=images,
        sections=sections,
        vlm_model=vlm_client_settings.model,
        vlm_prompt=vlm_parser_settings.prompt,
    )
    (out_dir / "preview.html").write_text(preview_html, encoding="utf-8")

    print(f"  → {len(pages)} pages, {len(sections)} sections")

    return {
        "name": pdf_path.name,
        "pages": len(pages),
        "sections": len(sections),
        "preview_path": f"{out_dir.name}/preview.html",
    }


def main() -> None:
    args = parse_args()

    # PDF 목록 수집
    pdfs: List[Path] = []
    if args.pdf:
        if not args.pdf.exists():
            raise FileNotFoundError(f"PDF not found: {args.pdf}")
        pdfs.append(args.pdf)
    elif args.input_dir:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        pdfs = list(args.input_dir.rglob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDF files found in {args.input_dir}")
    else:
        raise SystemExit("Either --pdf or --input-dir must be specified")

    print(f"Found {len(pdfs)} PDF(s) to process")

    # 출력 디렉토리 설정
    if args.no_timestamp:
        run_dir = args.output_dir
    else:
        run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # VLM 클라이언트 및 서비스 초기화
    vlm_client = OpenAIVisionClient(
        base_url=vlm_client_settings.base_url,
        model=vlm_client_settings.model,
        timeout=vlm_client_settings.timeout,
    )
    svc = DocumentIngestService.for_vlm(vlm_client=vlm_client)

    # 각 PDF 처리
    index_entries = []
    for pdf in pdfs:
        # 입력 디렉토리 기준 상대 경로 유지
        if args.input_dir:
            rel = pdf.relative_to(args.input_dir)
            out_dir = run_dir / rel.parent / pdf.stem
        else:
            out_dir = run_dir / pdf.stem

        try:
            entry = process_pdf(
            pdf_path=pdf,
            out_dir=out_dir,
            svc=svc,
            dpi=args.dpi,
            doc_type=args.doc_type,
            extract_figures=args.extract_figures,
            bbox_origin=args.bbox_origin,
        )
            # 상대 경로 수정
            entry["preview_path"] = str(out_dir.relative_to(run_dir) / "preview.html")
            index_entries.append(entry)
        except Exception as e:
            print(f"  ✗ Error processing {pdf}: {e}")
            continue

    # 인덱스 HTML 생성
    if index_entries:
        index_html = generate_index_html(run_dir, index_entries)
        (run_dir / "index.html").write_text(index_html, encoding="utf-8")

    print(f"\nDone! Open in browser: {run_dir / 'index.html'}")


if __name__ == "__main__":
    main()
