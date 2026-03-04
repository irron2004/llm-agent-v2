"""Phase 2 전체 chunking 오케스트레이션 스크립트.

5종 문서를 모두 chunking하여 all_chunks.jsonl로 출력한다.

Usage:
    python scripts/chunk_v3/run_chunking.py \
        --vlm-dir data/vlm_parsed \
        --myservice-dir /home/llm-share/datasets/pe_agent_data/pe_preprocess_data/myservice_txt \
        --gcb-path /home/llm-share/datasets/pe_agent_data/pe_preprocess_data/gcb_raw/20260126/scraped_gcb.json \
        --output data/chunks_v3/all_chunks.jsonl
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.chunkers import chunk_gcb, chunk_myservice, chunk_vlm_parsed
from scripts.chunk_v3.common import ChunkV3Document, save_chunks_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: 전체 문서 chunking")
    parser.add_argument(
        "--vlm-dir", default="data/vlm_parsed",
        help="VLM 파싱 결과 디렉토리 (기본: data/vlm_parsed)",
    )
    parser.add_argument(
        "--myservice-dir",
        default="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/myservice_txt",
        help="MyService TXT 디렉토리",
    )
    parser.add_argument(
        "--gcb-path",
        default="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/gcb_raw/20260126/scraped_gcb.json",
        help="GCB raw JSON 파일 경로",
    )
    parser.add_argument(
        "--output", default="data/chunks_v3/all_chunks.jsonl",
        help="출력 JSONL 파일 경로 (기본: data/chunks_v3/all_chunks.jsonl)",
    )
    parser.add_argument(
        "--gcb-chunk-size", type=int, default=512,
        help="GCB chunk 크기 (기본: 512)",
    )
    parser.add_argument(
        "--skip-vlm", action="store_true",
        help="VLM parsed 문서 건너뛰기 (MyService/GCB만 처리)",
    )
    parser.add_argument(
        "--skip-myservice", action="store_true",
        help="MyService 건너뛰기",
    )
    parser.add_argument(
        "--skip-gcb", action="store_true",
        help="GCB 건너뛰기",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_chunks: list[ChunkV3Document] = []
    stats: dict[str, int] = {}

    # VLM parsed 문서 (SOP PDF/PPTX, TS, Setup Manual)
    if not args.skip_vlm:
        for doc_type in ["sop_pdf", "sop_pptx", "ts", "setup_manual"]:
            pattern = str(Path(args.vlm_dir) / doc_type / "*.json")
            json_paths = sorted(glob(pattern))

            type_chunks: list[ChunkV3Document] = []
            for json_path in json_paths:
                type_chunks.extend(chunk_vlm_parsed(doc_type, json_path))

            all_chunks.extend(type_chunks)
            stats[doc_type] = len(type_chunks)
            print(f"  {doc_type}: {len(json_paths)} docs → {len(type_chunks)} chunks")

    # MyService
    if not args.skip_myservice:
        myservice_dir = Path(args.myservice_dir)
        if myservice_dir.exists():
            txt_files = sorted(myservice_dir.glob("*.txt"))
            ms_chunks: list[ChunkV3Document] = []

            for txt_path in txt_files:
                ms_chunks.extend(chunk_myservice(txt_path))

            all_chunks.extend(ms_chunks)
            stats["myservice"] = len(ms_chunks)
            print(f"  myservice: {len(txt_files)} files → {len(ms_chunks)} chunks")
        else:
            print(f"  myservice: directory not found ({myservice_dir})")

    # GCB
    if not args.skip_gcb:
        gcb_path = Path(args.gcb_path)
        if gcb_path.exists():
            gcb_chunks = chunk_gcb(
                gcb_path,
                chunk_size=args.gcb_chunk_size,
            )
            all_chunks.extend(gcb_chunks)
            stats["gcb"] = len(gcb_chunks)
            print(f"  gcb: → {len(gcb_chunks)} chunks")
        else:
            print(f"  gcb: file not found ({gcb_path})")

    # Save
    save_chunks_jsonl(args.output, all_chunks)

    # Summary
    print(f"\n{'='*60}")
    print("CHUNKING SUMMARY")
    print(f"  Output: {args.output}")
    total = 0
    for doc_type, count in stats.items():
        print(f"  {doc_type}: {count}")
        total += count
    print(f"  TOTAL: {total} chunks")


if __name__ == "__main__":
    main()
