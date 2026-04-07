"""Batch update existing ES chunks with `components` field.

Scans all chunks in chunk_v3_content index, extracts component names
using the component dictionary, and updates each chunk via bulk API.

Usage:
    # Dry run (report only, no updates)
    python -m backend.scripts.batch_update_components --dry-run

    # Execute update
    python -m backend.scripts.batch_update_components

    # Custom index
    python -m backend.scripts.batch_update_components --index chunk_v3_content
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from typing import Any

from elasticsearch import Elasticsearch, helpers

from backend.config.settings import search_settings
from backend.domain.component_dictionary import (
    get_component_dict,
    match_components,
    match_components_fuzzy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
SCROLL_TIMEOUT = "5m"

# Fields to fetch for matching
_SCAN_FIELDS = [
    "chunk_id",
    "content",
    "doc_type",
    "device_name",
    "doc_description",
    "components",
    "extra_meta",
]


def _ensure_mapping(es: Elasticsearch, index: str) -> None:
    """Add components field to mapping if not present."""
    mapping = es.indices.get_mapping(index=index)
    index_mapping = mapping.get(index, {})
    properties = index_mapping.get("mappings", {}).get("properties", {})

    if "components" not in properties:
        logger.info("Adding 'components' field to mapping of %s", index)
        es.indices.put_mapping(
            index=index,
            body={
                "properties": {
                    "components": {"type": "keyword", "doc_values": True},
                }
            },
        )
        logger.info("Mapping updated successfully")
    else:
        logger.info("'components' field already exists in %s mapping", index)


def _extract_doc_description(source: dict[str, Any]) -> str:
    """Extract doc_description from source, checking extra_meta as fallback."""
    desc = source.get("doc_description", "")
    if not desc:
        extra = source.get("extra_meta") or {}
        if isinstance(extra, dict):
            desc = extra.get("doc_description", "")
    return str(desc)


def run_batch_update(
    index: str,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Run batch component extraction and update.

    Args:
        index: Elasticsearch index name.
        dry_run: If True, only report what would be updated.
        skip_existing: Skip chunks that already have components.

    Returns:
        Dict with statistics.
    """
    es = Elasticsearch(hosts=[search_settings.es_host])
    if not es.ping():
        logger.error("Cannot connect to Elasticsearch at %s", search_settings.es_host)
        sys.exit(1)

    # Ensure mapping has components field
    if not dry_run:
        _ensure_mapping(es, index)

    comp_dict = get_component_dict()
    logger.info("Loaded component dictionary: %d entries", len(comp_dict))

    stats: Counter[str] = Counter()
    bulk_actions: list[dict[str, Any]] = []
    component_freq: Counter[str] = Counter()

    # Scan all documents
    query: dict[str, Any] = {"match_all": {}}
    if skip_existing:
        query = {
            "bool": {
                "must_not": {"exists": {"field": "components"}},
            }
        }

    logger.info("Scanning index %s (skip_existing=%s)...", index, skip_existing)

    for hit in helpers.scan(
        es,
        index=index,
        query={"query": query},
        _source=_SCAN_FIELDS,
        scroll=SCROLL_TIMEOUT,
        size=BATCH_SIZE,
    ):
        stats["scanned"] += 1
        doc_id = hit["_id"]
        source = hit.get("_source", {})

        # Skip if already has components and skip_existing
        existing = source.get("components", [])
        if existing and skip_existing:
            stats["skipped_existing"] += 1
            continue

        content = str(source.get("content", ""))
        doc_type = str(source.get("doc_type", ""))
        doc_description = _extract_doc_description(source)

        # Strategy based on doc_type
        if doc_type in ("myservice", "gcb"):
            # Fuzzy match on doc_description for user-written content
            components = match_components_fuzzy(
                doc_description, comp_dict, threshold=85
            )
            # Also try exact match on content as supplement
            if not components:
                components = match_components(content, comp_dict)
        else:
            # Exact match on content for structured documents (SOP, TS)
            components = match_components(content, comp_dict)

        if components:
            stats["updated"] += 1
            for c in components:
                component_freq[c] += 1

            if not dry_run:
                bulk_actions.append({
                    "_op_type": "update",
                    "_index": index,
                    "_id": doc_id,
                    "doc": {"components": components},
                })

                # Flush batch
                if len(bulk_actions) >= BATCH_SIZE:
                    _flush_bulk(es, bulk_actions, stats)
                    bulk_actions.clear()
        else:
            stats["no_match"] += 1

        # Progress logging
        if stats["scanned"] % 5000 == 0:
            logger.info(
                "Progress: scanned=%d, updated=%d, no_match=%d",
                stats["scanned"],
                stats["updated"],
                stats["no_match"],
            )

    # Flush remaining
    if bulk_actions and not dry_run:
        _flush_bulk(es, bulk_actions, stats)

    # Report
    logger.info("=" * 60)
    logger.info("Batch update %s", "DRY RUN" if dry_run else "COMPLETE")
    logger.info("  Scanned:  %d", stats["scanned"])
    logger.info("  Updated:  %d", stats["updated"])
    logger.info("  No match: %d", stats["no_match"])
    logger.info("  Skipped:  %d", stats.get("skipped_existing", 0))
    logger.info("  Errors:   %d", stats.get("bulk_errors", 0))
    logger.info("")
    logger.info("Top 20 components found:")
    for comp, count in component_freq.most_common(20):
        logger.info("  %-40s %5d", comp, count)

    return dict(stats)


def _flush_bulk(
    es: Elasticsearch,
    actions: list[dict[str, Any]],
    stats: Counter[str],
) -> None:
    """Execute bulk update and track errors."""
    try:
        success, errors = helpers.bulk(
            es, actions, raise_on_error=False, stats_only=False
        )
        if errors:
            stats["bulk_errors"] += len(errors)
            for err in errors[:3]:
                logger.warning("Bulk error: %s", err)
    except Exception as exc:
        logger.error("Bulk update failed: %s", exc)
        stats["bulk_errors"] += len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch update ES chunks with components field"
    )
    parser.add_argument(
        "--index",
        default=search_settings.v3_content_index or "chunk_v3_content",
        help="Elasticsearch content index name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only, do not update",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process chunks that already have components",
    )
    args = parser.parse_args()

    run_batch_update(
        index=args.index,
        dry_run=args.dry_run,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
