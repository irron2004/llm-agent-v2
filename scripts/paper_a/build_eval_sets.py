from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a.canonicalize import (
    compact_key,
    doc_id_variant_batch_sop,
    doc_id_variant_vlm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Paper A explicit eval set from SOP question CSV"
    )
    _ = parser.add_argument("--sop-csv", required=True, help="Path to SOP CSV file")
    _ = parser.add_argument(
        "--corpus-meta", required=True, help="Path to corpus doc_meta.jsonl"
    )
    _ = parser.add_argument("--out-dir", required=True, help="Output directory")
    return parser.parse_args()


def _normalize_header(name: str) -> str:
    compact = "".join(ch for ch in str(name).strip().lower() if ch not in {" ", "_"})
    return compact


def _find_column(
    fieldnames: list[str],
    *,
    preferred: str,
    fallbacks: list[str],
) -> str | None:
    normalized_to_original: dict[str, str] = {}
    for name in fieldnames:
        _ = normalized_to_original.setdefault(_normalize_header(name), name)

    for candidate in [preferred, *fallbacks]:
        found = normalized_to_original.get(_normalize_header(candidate))
        if found:
            return found
    return None


def _load_doc_meta(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    source_to_doc_id: dict[str, str] = {}
    doc_id_to_topic: dict[str, str] = {}
    topic_to_devices: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            loaded = cast(object, json.loads(raw))
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"Invalid doc_meta row at line {line_no}: expected object"
                )
            row = cast(dict[str, object], loaded)

            source_file = str(row.get("source_file") or "").strip()
            es_doc_id = str(row.get("es_doc_id") or "").strip()
            topic = str(row.get("topic") or "").strip()
            es_device_name = str(row.get("es_device_name") or "").strip()
            if not source_file or not es_doc_id:
                raise RuntimeError(
                    f"Invalid doc_meta row at line {line_no}: missing source_file/es_doc_id"
                )

            if (
                source_file in source_to_doc_id
                and source_to_doc_id[source_file] != es_doc_id
            ):
                raise RuntimeError(
                    f"Conflicting source_file mapping in doc_meta: {source_file!r}"
                )
            source_to_doc_id[source_file] = es_doc_id

            existing_topic = doc_id_to_topic.get(es_doc_id)
            if existing_topic is None or (not existing_topic and topic):
                doc_id_to_topic[es_doc_id] = topic

            compact_device = compact_key(es_device_name)
            if topic and compact_device:
                if topic not in topic_to_devices:
                    topic_to_devices[topic] = set()
                topic_to_devices[topic].add(compact_device)

    if not source_to_doc_id:
        raise RuntimeError(f"No rows loaded from corpus meta: {path}")

    topic_to_degree = {
        topic: len(devices) for topic, devices in topic_to_devices.items()
    }
    return source_to_doc_id, doc_id_to_topic, topic_to_degree


def _device_aliases(device: str) -> list[str]:
    base = str(device or "").strip()
    if not base:
        return []

    aliases: set[str] = {
        base,
        base.lower(),
        base.upper(),
        base.replace(" ", ""),
        base.replace("_", ""),
        base.replace(" ", "").replace("_", ""),
    }

    compact = compact_key(base)
    if compact:
        aliases.add(compact)

    cleaned = [alias.strip() for alias in aliases if alias.strip()]
    cleaned.sort(key=lambda item: (-len(item), item))
    return cleaned


def _mask_query(query: str, target_device: str) -> str:
    masked = str(query or "")
    aliases = _device_aliases(target_device)
    if not aliases:
        return masked

    for alias in aliases:
        alias_pattern = re.escape(alias)
        masked = re.sub(rf"(?i){alias_pattern}\s*설비", "[DEVICE]", masked)
        masked = re.sub(rf"(?i){alias_pattern}", "[DEVICE]", masked)

    masked = re.sub(r"\[DEVICE\](?:\s*\[DEVICE\])+", "[DEVICE]", masked)
    return masked


def _infer_topic(row: dict[str, object], doc_id_to_topic: dict[str, str]) -> str:
    row_topic = str(row.get("topic") or "").strip()
    if row_topic:
        return row_topic

    gold_doc_ids = row.get("gold_doc_ids")
    if not isinstance(gold_doc_ids, list):
        return ""
    for gold_doc_id in cast(list[object], gold_doc_ids):
        if not isinstance(gold_doc_id, str):
            continue
        topic = doc_id_to_topic.get(gold_doc_id, "")
        if topic:
            return topic
    return ""


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(row, ensure_ascii=False))
            _ = f.write("\n")


def _resolve_gold_doc_id(
    gold_source_file: str,
    source_to_doc_id: dict[str, str],
    corpus_doc_ids: set[str],
) -> tuple[str | None, str]:
    exact = source_to_doc_id.get(gold_source_file)
    if exact:
        return exact, "source_file_exact"

    stem = Path(gold_source_file).stem
    candidates: list[tuple[str, str]] = [
        ("doc_id_variant_vlm", doc_id_variant_vlm(stem)),
        ("doc_id_variant_batch_sop", doc_id_variant_batch_sop(stem)),
    ]
    for method, candidate in candidates:
        if candidate and candidate in corpus_doc_ids:
            return candidate, method

    return None, "unresolved"


def run() -> int:
    args = parse_args()
    sop_csv = Path(cast(str, args.sop_csv))
    corpus_meta = Path(cast(str, args.corpus_meta))
    out_dir = Path(cast(str, args.out_dir))

    if not sop_csv.exists() or not sop_csv.is_file():
        print(f"sop-csv file not found: {sop_csv}", file=sys.stderr)
        return 1
    if not corpus_meta.exists() or not corpus_meta.is_file():
        print(f"corpus-meta file not found: {corpus_meta}", file=sys.stderr)
        return 1

    try:
        source_to_doc_id, doc_id_to_topic, topic_to_degree = _load_doc_meta(corpus_meta)
        corpus_doc_ids = set(doc_id_to_topic.keys())

        with sop_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

            query_col = _find_column(
                fieldnames,
                preferred="질문내용",
                fallbacks=["질문 내용", "query", "question"],
            )
            device_col = _find_column(
                fieldnames,
                preferred="장비",
                fallbacks=["target_device", "device", "설비"],
            )
            gold_source_col = _find_column(
                fieldnames,
                preferred="정답문서",
                fallbacks=["정답 문서", "gold_source_file", "source_file"],
            )
            gold_pages_col = _find_column(
                fieldnames,
                preferred="정답 페이지",
                fallbacks=["정답페이지", "gold_pages", "pages", "page"],
            )

            missing_columns: list[str] = []
            if query_col is None:
                missing_columns.append("query(질문내용)")
            if device_col is None:
                missing_columns.append("target_device(장비)")
            if gold_source_col is None:
                missing_columns.append("gold source file(정답문서)")
            if gold_pages_col is None:
                missing_columns.append("expected pages(정답 페이지)")
            if missing_columns:
                print(
                    "Missing required columns: " + ", ".join(missing_columns),
                    file=sys.stderr,
                )
                return 1

            explicit_rows: list[dict[str, object]] = []
            unmatched_rows: list[dict[str, object]] = []

            for csv_line_no, row in enumerate(reader, start=2):
                query = str(row.get(query_col, "") or "").strip()
                target_device = str(row.get(device_col, "") or "").strip()
                gold_source_file = str(row.get(gold_source_col, "") or "").strip()
                gold_pages_raw = str(row.get(gold_pages_col, "") or "").strip()

                if not gold_source_file:
                    unmatched_row: dict[str, object] = {
                        "csv_line_no": csv_line_no,
                        "reason": "empty_gold_source_file",
                        "query": query,
                        "target_device": target_device,
                        "gold_source_file": gold_source_file,
                        "gold_pages": gold_pages_raw,
                    }
                    unmatched_rows.append(unmatched_row)
                    continue

                resolved_doc_id, resolution_method = _resolve_gold_doc_id(
                    gold_source_file,
                    source_to_doc_id,
                    corpus_doc_ids,
                )
                if not resolved_doc_id:
                    unresolved_row: dict[str, object] = {
                        "csv_line_no": csv_line_no,
                        "reason": "unresolved_gold_doc",
                        "resolution_method": resolution_method,
                        "query": query,
                        "target_device": target_device,
                        "gold_source_file": gold_source_file,
                        "gold_pages": gold_pages_raw,
                        "gold_source_stem": Path(gold_source_file).stem,
                        "doc_id_variant_vlm": doc_id_variant_vlm(
                            Path(gold_source_file).stem
                        ),
                        "doc_id_variant_batch_sop": doc_id_variant_batch_sop(
                            Path(gold_source_file).stem
                        ),
                    }
                    unmatched_rows.append(unresolved_row)
                    continue

                qid = f"A-E-{len(explicit_rows) + 1:04d}"
                output_row: dict[str, object] = {
                    "qid": qid,
                    "split": "explicit",
                    "query": query,
                    "target_device": target_device,
                    "gold_doc_ids": [resolved_doc_id],
                    "gold_source_files": [gold_source_file],
                    "gold_pages": [gold_pages_raw] if gold_pages_raw else [],
                }
                topic = doc_id_to_topic.get(resolved_doc_id, "")
                if topic:
                    output_row["topic"] = topic
                explicit_rows.append(output_row)

    except Exception as exc:
        print(f"build_eval_sets failed: {exc}", file=sys.stderr)
        return 1

    explicit_path = out_dir / "explicit.jsonl"
    masked_path = out_dir / "masked.jsonl"
    ambiguous_path = out_dir / "ambiguous.jsonl"
    unmatched_path = out_dir / "unmatched_gold.jsonl"

    if unmatched_rows:
        _write_jsonl(unmatched_path, unmatched_rows)
        print(
            (
                f"Failed to resolve {len(unmatched_rows)} rows out of "
                f"{len(explicit_rows) + len(unmatched_rows)}. See {unmatched_path}"
            ),
            file=sys.stderr,
        )
        return 1

    if unmatched_path.exists():
        unmatched_path.unlink()

    masked_rows: list[dict[str, object]] = []
    for explicit_row in explicit_rows:
        target_device = str(explicit_row.get("target_device") or "")
        masked_query = _mask_query(str(explicit_row.get("query") or ""), target_device)
        if target_device and compact_key(target_device) in compact_key(masked_query):
            raise RuntimeError(
                f"Masking failed for qid={explicit_row.get('qid')}: target device remained in query"
            )
        masked_row = dict(explicit_row)
        masked_row["qid"] = f"A-M-{len(masked_rows) + 1:04d}"
        masked_row["split"] = "masked"
        masked_row["query"] = masked_query
        masked_rows.append(masked_row)

    ambiguous_rows: list[dict[str, object]] = []
    for masked_row in masked_rows:
        topic = _infer_topic(masked_row, doc_id_to_topic)
        if topic_to_degree.get(topic, 0) < 2:
            continue
        ambiguous_row = dict(masked_row)
        ambiguous_row["qid"] = f"A-A-{len(ambiguous_rows) + 1:04d}"
        ambiguous_row["split"] = "ambiguous"
        ambiguous_rows.append(ambiguous_row)

    _write_jsonl(explicit_path, explicit_rows)
    _write_jsonl(masked_path, masked_rows)
    _write_jsonl(ambiguous_path, ambiguous_rows)
    print(f"Wrote {len(explicit_rows)} rows to {explicit_path}")
    print(f"Wrote {len(masked_rows)} rows to {masked_path}")
    print(f"Wrote {len(ambiguous_rows)} rows to {ambiguous_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
