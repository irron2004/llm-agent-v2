"""Elasticsearch index migration script for metadata fields.

Migrates from v1 (or current version) to v2 with new fields:
- device_name (keyword)
- doc_description (text)
- chapter (keyword)
- chunk_summary (text)

Usage:
    # Dry run (preview changes)
    python scripts/es_migrate_v2.py --dry-run

    # Execute migration
    python scripts/es_migrate_v2.py --execute

    # Force re-create even if target exists
    python scripts/es_migrate_v2.py --execute --force

    # Specify source/target versions
    python scripts/es_migrate_v2.py --execute --from-version 1 --to-version 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ES index migration for metadata fields")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-create target index even if it exists",
    )
    parser.add_argument(
        "--from-version",
        type=int,
        default=None,
        help="Source index version (default: current alias target)",
    )
    parser.add_argument(
        "--to-version",
        type=int,
        default=None,
        help="Target index version (default: from_version + 1)",
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Skip reindexing (just create new index)",
    )
    parser.add_argument(
        "--no-switch",
        action="store_true",
        help="Don't switch alias after migration",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old index after successful migration",
    )
    return parser.parse_args()


def get_doc_count(manager: EsIndexManager, version: int) -> int:
    """Get document count for an index version."""
    index_name = manager.get_index_name(version)
    try:
        result = manager.es.count(index=index_name)
        return result.get("count", 0)
    except Exception:
        return 0


def reindex_data(
    manager: EsIndexManager,
    from_version: int,
    to_version: int,
    *,
    wait_for_completion: bool = True,
) -> dict:
    """Reindex data from old index to new index."""
    source_index = manager.get_index_name(from_version)
    dest_index = manager.get_index_name(to_version)

    print(f"Reindexing {source_index} -> {dest_index}...")

    body = {
        "source": {"index": source_index},
        "dest": {"index": dest_index},
    }

    result = manager.es.reindex(
        body=body,
        wait_for_completion=wait_for_completion,
        request_timeout=3600,  # 1 hour timeout for large indices
    )

    return result


def main() -> None:
    args = parse_args()

    if not args.dry_run and not args.execute:
        print("ERROR: Specify either --dry-run or --execute")
        sys.exit(1)

    # Initialize manager
    manager = EsIndexManager(
        es_host=search_settings.es_host,
        env=search_settings.es_env,
        index_prefix=search_settings.es_index_prefix,
        es_user=search_settings.es_user or None,
        es_password=search_settings.es_password or None,
    )

    # Determine versions
    current_alias_target = manager.get_alias_target()
    latest_version = manager.get_latest_version() or 0

    if args.from_version is not None:
        from_version = args.from_version
    elif current_alias_target:
        # Parse version from alias target
        try:
            from_version = int(current_alias_target.split("_v")[-1])
        except (ValueError, IndexError):
            print(f"ERROR: Cannot parse version from alias target: {current_alias_target}")
            sys.exit(1)
    else:
        from_version = latest_version

    to_version = args.to_version if args.to_version is not None else from_version + 1

    # Print status
    print("=" * 60)
    print("ES Index Migration")
    print("=" * 60)
    print(f"Host: {search_settings.es_host}")
    print(f"Environment: {search_settings.es_env}")
    print(f"Index prefix: {search_settings.es_index_prefix}")
    print(f"Current alias target: {current_alias_target or 'None'}")
    print(f"Latest version: {latest_version or 'None'}")
    print()
    print(f"Source version: v{from_version}")
    print(f"Target version: v{to_version}")
    print()

    source_index = manager.get_index_name(from_version)
    target_index = manager.get_index_name(to_version)

    source_exists = manager.index_exists(from_version)
    target_exists = manager.index_exists(to_version)

    print(f"Source index: {source_index} (exists: {source_exists})")
    print(f"Target index: {target_index} (exists: {target_exists})")

    if source_exists:
        doc_count = get_doc_count(manager, from_version)
        print(f"Source document count: {doc_count:,}")
    else:
        doc_count = 0
        print("WARNING: Source index does not exist!")

    print()

    # Migration plan
    print("Migration Plan:")
    steps = []

    if target_exists and args.force:
        steps.append(f"1. Delete existing target index {target_index}")
    elif target_exists and not args.force:
        print(f"ERROR: Target index {target_index} already exists. Use --force to overwrite.")
        sys.exit(1)

    steps.append(f"{len(steps)+1}. Create new index {target_index} with updated mapping")

    if source_exists and not args.no_reindex:
        steps.append(f"{len(steps)+1}. Reindex {doc_count:,} documents from {source_index}")

    if not args.no_switch:
        steps.append(f"{len(steps)+1}. Switch alias to {target_index}")

    if args.delete_old and source_exists:
        steps.append(f"{len(steps)+1}. Delete old index {source_index}")

    for step in steps:
        print(f"  {step}")

    print()

    if args.dry_run:
        print("DRY RUN - No changes made")
        return

    # Execute migration
    print("Executing migration...")
    print()

    # Step 1: Delete target if force
    if target_exists and args.force:
        print(f"Deleting existing index {target_index}...")
        manager.delete_index(to_version)
        print("  Done")
        time.sleep(1)

    # Step 2: Create new index
    print(f"Creating index {target_index}...")
    manager.create_index(
        version=to_version,
        dims=search_settings.es_embedding_dims,
        skip_if_exists=False,
    )
    print("  Done")

    # Step 3: Reindex data
    if source_exists and not args.no_reindex and doc_count > 0:
        print(f"Reindexing {doc_count:,} documents...")
        start_time = time.time()
        result = reindex_data(manager, from_version, to_version)
        elapsed = time.time() - start_time

        created = result.get("created", 0)
        updated = result.get("updated", 0)
        failures = result.get("failures", [])

        print(f"  Created: {created:,}")
        print(f"  Updated: {updated:,}")
        print(f"  Failures: {len(failures)}")
        print(f"  Time: {elapsed:.1f}s")

        if failures:
            print("  WARNING: Some documents failed to reindex!")
            for f in failures[:5]:
                print(f"    - {f}")

        # Verify document count
        new_count = get_doc_count(manager, to_version)
        print(f"  Target document count: {new_count:,}")

    # Step 4: Switch alias
    if not args.no_switch:
        print(f"Switching alias to {target_index}...")
        manager.switch_alias(to_version)
        print("  Done")

    # Step 5: Delete old index
    if args.delete_old and source_exists:
        print(f"Deleting old index {source_index}...")
        manager.delete_index(from_version)
        print("  Done")

    print()
    print("=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)
    print()
    print("New mapping includes fields:")
    print("  - device_name (keyword)")
    print("  - doc_description (text)")
    print("  - chapter (keyword)")
    print("  - chunk_summary (text)")
    print()
    print("These fields will be populated during document ingestion.")
    print("Run vlm_es_ingest.py to re-ingest documents with metadata extraction.")


if __name__ == "__main__":
    main()
