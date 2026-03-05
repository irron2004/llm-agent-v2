from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from elasticsearch import NotFoundError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager
from scripts.paper_a.canonicalize import (
    compact_key,
    doc_id_variant_batch_sop,
    doc_id_variant_vlm,
    normalize_doc_type_es,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corpus metadata by joining normalize-table rows to ES doc_ids"
    )
    _ = parser.add_argument(
        "--normalize-table",
        type=str,
        required=True,
        help="Path to markdown normalize table",
    )
    _ = parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for corpus metadata files",
    )
    _ = parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Elasticsearch host override (default: SEARCH_ES_HOST)",
    )
    _ = parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Alias or index name (default: rag_chunks_{SEARCH_ES_ENV}_current)",
    )
    return parser.parse_args()


def resolve_alias_or_index(manager: EsIndexManager, requested: str) -> tuple[str, str]:
    default_alias = manager.get_alias_name()

    if requested == default_alias:
        target = manager.get_alias_target()
        if not target:
            raise RuntimeError(
                f"Alias '{requested}' does not exist or is not pointing to an index"
            )
        return requested, target

    try:
        alias_response = manager.es.indices.get_alias(name=requested)
        alias_result = cast(dict[str, object], cast(object, alias_response.body))
        alias_targets = list(alias_result.keys())
        if not alias_targets:
            raise RuntimeError(f"Alias '{requested}' exists but has no target index")
        return requested, alias_targets[0]
    except NotFoundError:
        if manager.es.indices.exists(index=requested):
            return requested, requested
        raise RuntimeError(f"Alias/index '{requested}' was not found") from None


def parse_section_doc_type(line: str) -> str | None:
    matched = re.match(r"^##\s+(.+?)\s*\(", line.strip())
    if not matched:
        return None
    return matched.group(1).strip()


def split_markdown_row(line: str) -> list[str]:
    raw = line.strip()
    if not raw.startswith("|") or "|" not in raw[1:]:
        return []
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) < 3:
        return []
    return parts[1:-1]


def is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells)


def _normalize_header_name(name: str) -> str:
    return re.sub(r"\s+", "", name.strip().lower())


def parse_normalize_table(path: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    current_doc_type = ""
    headers: list[str] = []
    source_idx = 0
    device_idx = 1
    work_type_idx = 2
    module_idx = 3
    topic_idx = 4

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            section_doc_type = parse_section_doc_type(line)
            if section_doc_type is not None:
                current_doc_type = section_doc_type
                headers = []
                continue

            cells = split_markdown_row(line)
            if not cells:
                continue

            if is_separator_row(cells):
                continue

            if not headers:
                headers = cells
                header_map = {
                    _normalize_header_name(name): idx
                    for idx, name in enumerate(headers)
                }
                source_idx = header_map.get(
                    "원본파일명", header_map.get("source_file", 0)
                )
                device_idx = header_map.get("device_name", 1)
                work_type_idx = header_map.get("work_type", 2)
                module_idx = header_map.get("module", 3)
                topic_idx = header_map.get("topic", 4)
                continue

            if len(cells) < len(headers):
                cells = cells + [""] * (len(headers) - len(cells))

            source_file = cells[source_idx].strip() if source_idx < len(cells) else ""
            if not source_file:
                raise RuntimeError(f"Empty source file cell at line {line_no}")

            rows.append(
                {
                    "row_no": len(rows) + 1,
                    "line_no": line_no,
                    "section_doc_type": current_doc_type,
                    "source_file": source_file,
                    "device_name": cells[device_idx].strip()
                    if device_idx < len(cells)
                    else "",
                    "work_type": cells[work_type_idx].strip()
                    if work_type_idx < len(cells)
                    else "",
                    "module": cells[module_idx].strip()
                    if module_idx < len(cells)
                    else "",
                    "topic": cells[topic_idx].strip() if topic_idx < len(cells) else "",
                }
            )

    return rows


def doc_id_exists(manager: EsIndexManager, index_name: str, doc_id: str) -> bool:
    response = manager.es.search(
        index=index_name,
        body={
            "size": 0,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "should": [
                        {"term": {"doc_id.keyword": doc_id}},
                        {"term": {"doc_id": doc_id}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        },
    )
    payload = cast(dict[str, object], cast(object, response.body))
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        return False
    total = hits.get("total")
    if not isinstance(total, dict):
        return False
    value = total.get("value")
    return isinstance(value, int) and value > 0


def strip_known_suffixes(doc_id: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    current = doc_id.strip("_")

    while current and current not in seen:
        seen.add(current)
        values.append(current)

        next_value = re.sub(r"_(?:r|rev|v)\d+$", "", current)
        next_value = re.sub(r"_(?:en|eng|jp|kr|kor|cn|ch)$", "", next_value)
        next_value = next_value.rstrip("_")
        if next_value == current:
            break
        current = next_value

    return values


def find_doc_ids_by_prefix(
    manager: EsIndexManager,
    index_name: str,
    prefix: str,
) -> list[str]:
    if len(prefix) < 8:
        return []

    response = manager.es.search(
        index=index_name,
        body={
            "size": 0,
            "query": {"prefix": {"doc_id": prefix}},
            "aggs": {
                "doc_ids": {
                    "terms": {
                        "field": "doc_id",
                        "size": 50,
                    }
                }
            },
        },
    )
    payload = cast(dict[str, object], cast(object, response.body))
    aggregations = payload.get("aggregations")
    if not isinstance(aggregations, dict):
        return []
    doc_ids = aggregations.get("doc_ids")
    if not isinstance(doc_ids, dict):
        return []
    buckets = doc_ids.get("buckets")
    if not isinstance(buckets, list):
        return []
    keys: list[str] = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        key = bucket.get("key")
        if isinstance(key, str) and key:
            keys.append(key)
    return sorted(set(keys))


def resolve_doc_id(
    manager: EsIndexManager,
    index_name: str,
    vlm_variant: str,
    batch_variant: str,
    expected_device_name: str,
    expected_doc_type: str,
    expected_topic: str,
    source_file_stem: str,
    exact_cache: dict[str, bool],
    prefix_cache: dict[str, list[str]],
    representative_cache: dict[str, dict[str, object] | None],
) -> tuple[str | None, str]:
    ordered = [vlm_variant, batch_variant]
    exact_hits: list[str] = []
    for candidate in ordered:
        if candidate not in exact_cache:
            exact_cache[candidate] = doc_id_exists(manager, index_name, candidate)
        if exact_cache[candidate]:
            exact_hits.append(candidate)

    if exact_hits:
        if len(exact_hits) == 1:
            return exact_hits[0], "vlm" if exact_hits[0] == vlm_variant else "batch_sop"
        if exact_hits[0] == exact_hits[1]:
            return exact_hits[0], "both_same"
        return exact_hits[0], "both_exist_prefer_vlm"

    fallback_candidates: list[str] = []
    for candidate in ordered:
        for stripped in strip_known_suffixes(candidate):
            if stripped not in fallback_candidates:
                fallback_candidates.append(stripped)

    for candidate in fallback_candidates:
        if candidate not in exact_cache:
            exact_cache[candidate] = doc_id_exists(manager, index_name, candidate)
        if exact_cache[candidate]:
            return candidate, "suffix_stripped_exact"

    prefix_hits: list[str] = []
    for candidate in fallback_candidates:
        if candidate not in prefix_cache:
            prefix_cache[candidate] = find_doc_ids_by_prefix(
                manager, index_name, candidate
            )
        for resolved in prefix_cache[candidate]:
            if resolved not in prefix_hits:
                prefix_hits.append(resolved)

    if len(prefix_hits) == 1:
        return prefix_hits[0], "prefix_unique"

    if prefix_hits:
        target_doc_type = normalize_doc_type_es(expected_doc_type)
        target_device = compact_key(expected_device_name)
        scored: list[tuple[int, str]] = []
        for candidate_doc_id in prefix_hits:
            if candidate_doc_id not in representative_cache:
                representative_cache[candidate_doc_id] = fetch_representative_source(
                    manager, index_name, candidate_doc_id
                )
            representative = representative_cache[candidate_doc_id] or {}
            score = 0
            if target_doc_type and representative.get("doc_type") == target_doc_type:
                score += 2
            rep_device = compact_key(
                cast(str | None, representative.get("device_name"))
            )
            if target_device and rep_device == target_device:
                score += 2
            scored.append((score, candidate_doc_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        if scored and scored[0][0] > 0:
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return scored[0][1], "prefix_scored"

    metadata_query_terms = [
        vlm_variant,
        batch_variant,
        expected_topic,
        source_file_stem,
    ]
    metadata_query = " ".join(term for term in metadata_query_terms if term).strip()
    if metadata_query:
        soft_clauses: list[dict[str, object]] = []
        target_doc_type = normalize_doc_type_es(expected_doc_type)
        if target_doc_type:
            soft_clauses.append({"term": {"doc_type": target_doc_type}})
        if expected_device_name:
            soft_clauses.append({"term": {"device_name": expected_device_name}})
            soft_clauses.append({"match": {"device_name": expected_device_name}})

        response = manager.es.search(
            index=index_name,
            body={
                "size": 50,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": metadata_query,
                                    "fields": [
                                        "doc_id^5",
                                        "search_text",
                                        "content",
                                    ],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "should": soft_clauses,
                        "minimum_should_match": 0,
                    }
                },
                "_source": ["doc_id"],
            },
        )
        payload = cast(dict[str, object], cast(object, response.body))
        hits_section = payload.get("hits")
        if not isinstance(hits_section, dict):
            return None, "metadata_search_no_hits"
        hits = hits_section.get("hits")
        if not isinstance(hits, list):
            return None, "metadata_search_no_hits"
        seen_doc_ids: list[str] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            source_obj = hit.get("_source")
            if not isinstance(source_obj, dict):
                continue
            source = cast(dict[str, object], source_obj)
            doc_id = source.get("doc_id")
            if isinstance(doc_id, str) and doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.append(doc_id)
        if seen_doc_ids:
            return seen_doc_ids[0], "metadata_search"

    return None, "not_found"


def fetch_representative_source(
    manager: EsIndexManager,
    index_name: str,
    doc_id: str,
) -> dict[str, object] | None:
    response = manager.es.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "should": [
                        {"term": {"doc_id.keyword": doc_id}},
                        {"term": {"doc_id": doc_id}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": 20,
            "sort": [{"page": {"order": "asc", "missing": "_last"}}],
            "_source": [
                "doc_id",
                "chunk_id",
                "device_name",
                "doc_type",
                "equip_id",
                "page",
            ],
        },
    )
    payload = cast(dict[str, object], cast(object, response.body))
    hits_section = payload.get("hits")
    if not isinstance(hits_section, dict):
        return None
    hits = hits_section.get("hits")
    if not isinstance(hits, list) or not hits:
        return None
    sources: list[dict[str, object]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source_obj = hit.get("_source")
        if isinstance(source_obj, dict):
            sources.append(cast(dict[str, object], source_obj))
    if not sources:
        return None
    numeric_pages: list[int] = []
    for source in sources:
        page_value = source.get("page")
        if isinstance(page_value, (int, float)):
            numeric_pages.append(int(page_value))
    page_sources = sources
    if numeric_pages:
        min_page = min(numeric_pages)
        page_sources = []
        for source in sources:
            page_value = source.get("page")
            if isinstance(page_value, (int, float)) and int(page_value) == min_page:
                page_sources.append(source)
    chosen = min(page_sources, key=lambda item: str(item.get("chunk_id") or ""))
    return chosen


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        _ = f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            _ = f.write("\n")


def build_snapshot(doc_meta_rows: list[dict[str, object]]) -> dict[str, object]:
    device_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()
    doc_type_counter: Counter[str] = Counter()

    for row in doc_meta_rows:
        device_name = str(row.get("es_device_name") or "")
        doc_type = str(row.get("es_doc_type") or "")
        topic = str(row.get("topic") or "")

        if device_name:
            device_counter[device_name] += 1
        if doc_type:
            doc_type_counter[doc_type] += 1
        if topic:
            topic_counter[topic] += 1

    return {
        "total_docs": len(doc_meta_rows),
        "counts": {
            "devices": dict(sorted(device_counter.items())),
            "doc_types": dict(sorted(doc_type_counter.items())),
            "topics": dict(sorted(topic_counter.items())),
        },
    }


def run() -> int:
    args = parse_args()
    normalize_table_path = Path(cast(str, args.normalize_table))
    out_dir = Path(cast(str, args.out_dir))
    host_arg = cast(str | None, args.host)
    index_arg = cast(str | None, args.index)

    if not normalize_table_path.exists() or not normalize_table_path.is_file():
        print(
            f"normalize-table file not found: {normalize_table_path}",
            file=sys.stderr,
        )
        return 1

    host = host_arg or search_settings.es_host
    env = search_settings.es_env
    index_prefix = search_settings.es_index_prefix
    alias_or_index = index_arg or f"{index_prefix}_{env}_current"

    manager = EsIndexManager(
        es_host=host,
        env=env,
        index_prefix=index_prefix,
        es_user=search_settings.es_user or None,
        es_password=search_settings.es_password or None,
        verify_certs=True,
    )

    try:
        if not manager.es.ping():
            raise RuntimeError(f"Cannot connect to Elasticsearch at {host}")

        _, resolved_index = resolve_alias_or_index(manager, alias_or_index)
        table_rows = parse_normalize_table(normalize_table_path)
        if not table_rows:
            raise RuntimeError(f"No table rows parsed from {normalize_table_path}")

        doc_meta_rows: list[dict[str, object]] = []
        unresolved: list[dict[str, object]] = []
        doc_id_list: list[str] = []
        exact_cache: dict[str, bool] = {}
        prefix_cache: dict[str, list[str]] = {}
        representative_cache: dict[str, dict[str, object] | None] = {}

        for row in table_rows:
            source_file = str(row["source_file"])
            file_stem = Path(source_file).stem
            vlm_variant = doc_id_variant_vlm(file_stem)
            batch_variant = doc_id_variant_batch_sop(file_stem)

            resolved_doc_id, _resolution = resolve_doc_id(
                manager,
                resolved_index,
                vlm_variant,
                batch_variant,
                str(row["device_name"]),
                str(row["section_doc_type"]),
                str(row["topic"]),
                file_stem,
                exact_cache,
                prefix_cache,
                representative_cache,
            )
            if not resolved_doc_id:
                unresolved.append(
                    {
                        "row_no": row["row_no"],
                        "line_no": row["line_no"],
                        "source_file": source_file,
                        "source_file_stem": file_stem,
                        "doc_id_variant_vlm": vlm_variant,
                        "doc_id_variant_batch_sop": batch_variant,
                        "reason": "neither_variant_found_in_es",
                    }
                )
                continue

            source = fetch_representative_source(
                manager, resolved_index, resolved_doc_id
            )
            if source is None:
                unresolved.append(
                    {
                        "row_no": row["row_no"],
                        "line_no": row["line_no"],
                        "source_file": source_file,
                        "source_file_stem": file_stem,
                        "resolved_doc_id": resolved_doc_id,
                        "reason": "doc_id_found_in_terms_but_no_representative_hit",
                    }
                )
                continue

            doc_meta_rows.append(
                {
                    "source_file": source_file,
                    "topic": str(row["topic"]),
                    "manifest_doc_type": str(row["section_doc_type"]),
                    "es_doc_id": resolved_doc_id,
                    "es_doc_type": normalize_doc_type_es(
                        cast(str | None, source.get("doc_type"))
                    ),
                    "es_device_name": cast(str | None, source.get("device_name")) or "",
                    "es_equip_id": cast(str | None, source.get("equip_id")) or "",
                }
            )
            doc_id_list.append(resolved_doc_id)

        unresolved_path = out_dir / "unresolved_docs.jsonl"
        if unresolved:
            write_jsonl(unresolved_path, unresolved)
            print(
                (
                    f"Failed to resolve {len(unresolved)} / {len(table_rows)} rows. "
                    f"See {unresolved_path}"
                ),
                file=sys.stderr,
            )
            return 1

        if unresolved_path.exists():
            unresolved_path.unlink()

        doc_meta_path = out_dir / "doc_meta.jsonl"
        doc_ids_path = out_dir / "corpus_doc_ids.txt"
        snapshot_path = out_dir / "corpus_snapshot.json"

        write_jsonl(doc_meta_path, doc_meta_rows)

        doc_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with doc_ids_path.open("w", encoding="utf-8") as f:
            _ = f.write("\n".join(doc_id_list))
            _ = f.write("\n")

        snapshot = build_snapshot(doc_meta_rows)
        write_json(snapshot_path, snapshot)

        print(
            f"Wrote {len(doc_meta_rows)} docs to {doc_meta_path} and {doc_ids_path}; snapshot={snapshot_path}"
        )
        return 0
    except Exception as exc:
        print(
            (
                "build_corpus_meta failed: "
                f"{exc}. host={host} env={env} alias_or_index={alias_or_index}"
            ),
            file=sys.stderr,
        )
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
