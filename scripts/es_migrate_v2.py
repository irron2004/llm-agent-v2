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


def verify_nori_plugin(manager: EsIndexManager) -> bool:
    """Verify that Nori plugin is installed in Elasticsearch."""
    try:
        response = manager.es.cat.plugins(format="json")
        for plugin in response:
            if plugin.get("component") == "analysis-nori":
                version = plugin.get("version", "unknown")
                print(f"[✓] Nori plugin verified: analysis-nori v{version}")
                return True
        print("[✗] Nori plugin NOT found")
        return False
    except Exception as e:
        print(f"[✗] Error checking plugins: {e}")
        return False


def test_nori_analyzer(manager: EsIndexManager, version: int) -> bool:
    """Test Nori analyzer on the target index with sample Korean text."""
    index_name = manager.get_index_name(version)
    test_text = "한국어 형태소 분석 테스트"

    try:
        response = manager.es.indices.analyze(
            index=index_name,
            body={"analyzer": "nori", "text": test_text}
        )
        tokens = response.get("tokens", [])
        if len(tokens) > 0:
            token_text = ", ".join([t["token"] for t in tokens[:5]])
            print(f"[✓] Nori analyzer working: '{test_text}' → [{token_text}...]")
            return True
        else:
            print("[✗] Nori analyzer returned no tokens")
            return False
    except Exception as e:
        print(f"[✗] Error testing Nori analyzer: {e}")
        return False


def validate_embeddings_preserved(
    manager: EsIndexManager,
    from_version: int,
    to_version: int,
    sample_size: int = 10
) -> bool:
    """Validate that embedding vectors are preserved during reindex."""
    source_index = manager.get_index_name(from_version)
    target_index = manager.get_index_name(to_version)

    print(f"[→] Validating embeddings preserved (sample size: {sample_size})...")

    try:
        # Get random sample from source
        sample_query = {
            "size": sample_size,
            "query": {
                "function_score": {
                    "query": {"match_all": {}},
                    "random_score": {}
                }
            },
            "_source": ["embedding"]
        }

        source_docs = manager.es.search(index=source_index, body=sample_query)
        hits = source_docs.get("hits", {}).get("hits", [])

        if not hits:
            print("[✗] No documents found in source index")
            return False

        mismatches = 0
        for hit in hits:
            doc_id = hit["_id"]
            source_embedding = hit["_source"].get("embedding")

            if not source_embedding:
                continue

            # Get same document from target
            try:
                target_doc = manager.es.get(index=target_index, id=doc_id)
                target_embedding = target_doc["_source"].get("embedding")

                # Compare embeddings
                if source_embedding != target_embedding:
                    mismatches += 1
                    print(f"[✗] Embedding mismatch for doc {doc_id}")

            except Exception as e:
                print(f"[✗] Error getting doc {doc_id} from target: {e}")
                mismatches += 1

        if mismatches == 0:
            print(f"[✓] All {len(hits)} sampled embeddings preserved correctly")
            return True
        else:
            print(f"[✗] {mismatches}/{len(hits)} embeddings have mismatches")
            return False

    except Exception as e:
        print(f"[✗] Error validating embeddings: {e}")
        return False


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
    parser.add_argument(
        "--verify-nori",
        action="store_true",
        help="Verify Nori plugin is installed before migration",
    )
    parser.add_argument(
        "--test-analyzer",
        action="store_true",
        help="Test Nori analyzer after index creation",
    )
    parser.add_argument(
        "--validate-embeddings",
        action="store_true",
        help="Validate embeddings are preserved during reindex",
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

    # Verify Nori plugin if requested
    if args.verify_nori:
        if not verify_nori_plugin(manager):
            print("ERROR: Nori plugin verification failed!")
            if not args.dry_run:
                print("Please ensure Nori plugin is installed before proceeding.")
                sys.exit(1)
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

    # Test Nori analyzer if requested
    if args.test_analyzer:
        print()
        test_nori_analyzer(manager, to_version)

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

        # Validate embeddings if requested
        if args.validate_embeddings:
            print()
            if not validate_embeddings_preserved(manager, from_version, to_version):
                print("ERROR: Embedding validation failed!")
                print("Aborting migration. Target index has been created but alias not switched.")
                sys.exit(1)

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
