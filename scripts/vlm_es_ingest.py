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

Prerequisites:
- vLLM (OpenAI-compatible) server running with a vision model (e.g., Qwen3-VL)
- Elasticsearch with rag_chunks_* index/alias created
- .env configured (VLM_CLIENT_*, SEARCH_*, RAG_* settings)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

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
    return parser.parse_args()


def to_section_objs(sections: Iterable[dict]) -> List[Section]:
    objs: List[Section] = []
    for sec in sections:
        objs.append(
            Section(
                title=sec.get("title", ""),
                text=sec.get("text", ""),
                page_start=sec.get("page_start"),
                page_end=sec.get("page_end"),
                metadata=sec.get("metadata", {}),
            )
        )
    return objs


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

    section_objs = to_section_objs(sections)
    print(f"Parsed sections: {len(section_objs)}")
    for i, sec in enumerate(section_objs[:3], 1):
        print(f"[Section {i}] {sec.title}\n{sec.text[:200]}...\n")

    # 3) ES ingest
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


if __name__ == "__main__":
    main()
