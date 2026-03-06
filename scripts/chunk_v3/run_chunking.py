from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.chunkers import chunk_gcb, chunk_myservice, chunk_vlm_parsed
from scripts.chunk_v3.common import (
    ChunkV3Document,
    canonicalize_doc_type,
    load_vlm_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: 전체 문서 chunking")
    parser.add_argument("--vlm-dir", default="data/vlm_parsed")
    parser.add_argument(
        "--myservice-dir",
        default="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/myservice_txt",
    )
    parser.add_argument(
        "--gcb-path",
        default="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/gcb_raw/20260126/scraped_gcb.json",
    )
    parser.add_argument("--output", default="data/chunks_v3/all_chunks.jsonl")
    parser.add_argument("--manifest", default="data/chunk_v3_manifest.json")
    parser.add_argument("--gcb-chunk-size", type=int, default=512)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-myservice", action="store_true")
    parser.add_argument("--skip-gcb", action="store_true")
    parser.add_argument("--validate-vlm", action="store_true")
    parser.add_argument(
        "--source-dir",
        default="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data",
    )
    parser.add_argument(
        "--validation-output", default="data/vlm_parsed/validation_report.json"
    )
    parser.add_argument("--stats-path", default="data/chunks_v3/chunking_stats.json")
    parser.add_argument("--stats-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()

    if args.validate_vlm and not args.skip_vlm:
        validate_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "chunk_v3" / "validate_vlm.py"),
            "--parsed-dir",
            str(args.vlm_dir),
            "--source-dir",
            str(args.source_dir),
            "--output",
            str(args.validation_output),
            "--fail-on-mismatch",
        ]
        subprocess.run(validate_cmd, check=True)

    stats: dict[str, dict[str, int]] = {}

    output_fp = None
    if not args.stats_only:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fp = output_path.open("w", encoding="utf-8")

    def ensure_bucket(doc_type: str) -> dict[str, int]:
        if doc_type not in stats:
            stats[doc_type] = {
                "input_docs": 0,
                "parsed_units": 0,
                "output_chunks": 0,
                "skipped_empty_chunks": 0,
            }
        return stats[doc_type]

    def emit(doc_type: str, chunks: list[ChunkV3Document]) -> None:
        bucket = ensure_bucket(doc_type)
        bucket["output_chunks"] += len(chunks)
        if output_fp is None:
            return
        for chunk in chunks:
            output_fp.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    if not args.skip_vlm:
        vlm_doc_type_dirs = {
            "sop": ["sop"],
            "ts": ["ts", "trouble_shooting", "troubleshooting"],
            "setup": ["setup", "setup_manual", "set_up_manual"],
        }
        for canonical_doc_type, alias_dirs in vlm_doc_type_dirs.items():
            seen: set[str] = set()
            json_paths: list[str] = []
            for alias_dir in alias_dirs:
                for path in sorted(
                    glob(str(Path(args.vlm_dir) / alias_dir / "*.json"))
                ):
                    if path not in seen:
                        seen.add(path)
                        json_paths.append(path)

            input_docs = len(json_paths)
            parsed_units = 0
            output_chunks = 0
            for json_path in json_paths:
                parsed = load_vlm_result(json_path)
                parsed_units += len(parsed.get("pages", []))
                chunks = chunk_vlm_parsed(
                    canonicalize_doc_type(canonical_doc_type),
                    json_path,
                    manifest_path=args.manifest,
                )
                emit(canonical_doc_type, chunks)
                output_chunks += len(chunks)

            bucket = ensure_bucket(canonical_doc_type)
            bucket["input_docs"] += input_docs
            bucket["parsed_units"] += parsed_units
            print(
                f"  {canonical_doc_type}: {input_docs} docs ({', '.join(alias_dirs)})"
                f" -> {output_chunks} chunks"
            )

    if not args.skip_myservice:
        myservice_dir = Path(args.myservice_dir)
        if myservice_dir.exists():
            txt_files = sorted(myservice_dir.glob("*.txt"))
            total_chunks = 0
            for txt_path in txt_files:
                chunks = chunk_myservice(txt_path)
                emit("myservice", chunks)
                total_chunks += len(chunks)

            bucket = ensure_bucket("myservice")
            bucket["input_docs"] += len(txt_files)
            bucket["parsed_units"] += len(txt_files)
            print(f"  myservice: {len(txt_files)} files -> {total_chunks} chunks")
        else:
            print(f"  myservice: directory not found ({myservice_dir})")

    if not args.skip_gcb:
        gcb_path = Path(args.gcb_path)
        if gcb_path.exists():
            raw_count = 0
            try:
                raw_data = json.loads(gcb_path.read_text(encoding="utf-8"))
                if isinstance(raw_data, list):
                    raw_count = len(raw_data)
                else:
                    raw_count = 1
            except Exception:
                raw_count = 0

            gcb_chunks = chunk_gcb(gcb_path, chunk_size=args.gcb_chunk_size)
            emit("gcb", gcb_chunks)
            bucket = ensure_bucket("gcb")
            bucket["input_docs"] += raw_count
            bucket["parsed_units"] += raw_count
            print(f"  gcb: -> {len(gcb_chunks)} chunks")
        else:
            print(f"  gcb: file not found ({gcb_path})")

    if output_fp is not None:
        output_fp.close()

    total_inputs = sum(v["input_docs"] for v in stats.values())
    total_parsed = sum(v["parsed_units"] for v in stats.values())
    total_chunks = sum(v["output_chunks"] for v in stats.values())

    stats_payload = {
        "raw_input_count": total_inputs,
        "parsed_unit_count": total_parsed,
        "chunk_count": total_chunks,
        "by_doc_type": stats,
        "runtime_seconds": round(time.time() - started_at, 3),
    }

    stats_path = Path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print("CHUNKING SUMMARY")
    if not args.stats_only:
        print(f"  Output: {args.output}")
    print(f"  Stats: {args.stats_path}")
    for doc_type, detail in stats.items():
        print(f"  {doc_type}: {detail['output_chunks']}")
    print(f"  TOTAL: {total_chunks} chunks")


if __name__ == "__main__":
    main()
