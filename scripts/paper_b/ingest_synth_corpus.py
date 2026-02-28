#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.es_ingest_service import EsIngestService
from backend.services.ingest.document_ingest_service import Section


DEFAULT_CORPUS = "data/synth_benchmarks/stability_bench_v1/corpus.jsonl"
DEFAULT_REPORT = ".sisyphus/evidence/paper-b/task-5-ingest-summary.json"


@dataclass(frozen=True)
class CliArgs:
    corpus: Path
    report: Path


def _to_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _to_metadata(row: dict[str, object]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    device_name = row.get("device_name")
    if device_name is not None:
        metadata["device_name"] = str(device_name)

    equip_id = row.get("equip_id")
    if equip_id not in (None, ""):
        metadata["equip_id"] = str(equip_id)

    chapter = row.get("chapter")
    if chapter not in (None, ""):
        metadata["chapter"] = str(chapter)

    return metadata


def _build_section(row: dict[str, object]) -> Section:
    text = str(row.get("content", ""))
    title = str(row.get("chapter") or row.get("doc_id") or "synthetic_doc")
    return Section(
        title=title,
        text=text,
        page_start=0,
        page_end=0,
        metadata=_to_metadata(row),
    )


def _to_path(value: object, field_name: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise RuntimeError(f"{field_name} must be a path")


def _parse_args() -> CliArgs:
    class _ArgsNamespace(argparse.Namespace):
        corpus: Path = Path(DEFAULT_CORPUS)
        report: Path = Path(DEFAULT_REPORT)

    parser = argparse.ArgumentParser(
        description="Ingest synthetic corpus JSONL into Elasticsearch"
    )
    _ = parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(DEFAULT_CORPUS),
        help=f"Corpus JSONL path (default: {DEFAULT_CORPUS})",
    )
    _ = parser.add_argument(
        "--report",
        type=Path,
        default=Path(DEFAULT_REPORT),
        help=f"Summary report path (default: {DEFAULT_REPORT})",
    )
    parsed = parser.parse_args(namespace=_ArgsNamespace())
    return CliArgs(
        corpus=_to_path(cast(object, parsed.corpus), "corpus"),
        report=_to_path(cast(object, parsed.report), "report"),
    )


def main() -> int:
    args = _parse_args()
    corpus_path = args.corpus
    report_path = args.report

    if not corpus_path.exists():
        print(f"FAIL: corpus file not found: {corpus_path}")
        return 1

    svc = EsIngestService.from_settings()

    docs_processed = 0
    chunks_indexed = 0
    failures: list[dict[str, str]] = []

    with corpus_path.open("r", encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line:
                continue

            doc_id_for_error = ""
            try:
                row_obj: object = json.loads(line)
                if not isinstance(row_obj, dict):
                    raise ValueError("row must be a JSON object")
                row = {str(k): v for k, v in row_obj.items()}

                doc_id = str(row.get("doc_id") or f"synth_row_{line_number:06d}")
                doc_id_for_error = doc_id
                doc_type = str(row.get("doc_type") or "synthetic")
                doc_tags = _to_tags(row.get("tags"))
                section = _build_section(row)

                result = svc.ingest_sections(
                    base_doc_id=doc_id,
                    sections=[section],
                    doc_type=doc_type,
                    tags=doc_tags,
                    refresh=False,
                )

                docs_processed += 1
                indexed = result.get("indexed", 0)
                chunks_indexed += int(indexed) if isinstance(indexed, int) else 0
            except Exception as exc:
                failures.append(
                    {
                        "line": str(line_number),
                        "doc_id": doc_id_for_error,
                        "error": str(exc),
                    }
                )

    _ = svc.es.indices.refresh(index=svc.index)

    report = {
        "corpus_path": str(corpus_path),
        "index": svc.index,
        "docs_processed": docs_processed,
        "chunks_indexed": chunks_indexed,
        "failures_count": len(failures),
        "failures": failures,
        "refreshed": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Ingested docs: {docs_processed}")
    print(f"Indexed chunks: {chunks_indexed}")
    print(f"Failures: {len(failures)}")
    print(f"Report: {report_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
