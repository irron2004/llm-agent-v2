#!/usr/bin/env python3
"""Migrate existing index to alias-based naming strategy.

This script migrates from direct index usage to alias-based rolling update strategy:
- Current: rag_chunks_dev_current (direct index)
- Target: rag_chunks_dev_v1 (versioned index) + rag_chunks_dev_current (alias)

Usage:
    # Dry run (preview changes)
    python scripts/migrate_to_alias_strategy.py --dry-run

    # Execute migration
    python scripts/migrate_to_alias_strategy.py

    # With custom settings
    python scripts/migrate_to_alias_strategy.py --es-host http://localhost:9200 --env dev
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch import EsIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def migrate_to_alias_strategy(
    es_host: str,
    env: str,
    index_prefix: str = "rag_chunks",
    target_version: int = 1,
    dry_run: bool = False,
) -> bool:
    """Migrate from direct index to alias-based strategy.

    Args:
        es_host: Elasticsearch host URL
        env: Environment name (dev, staging, prod)
        index_prefix: Index prefix (default: rag_chunks)
        target_version: Target version number (default: 1)
        dry_run: If True, only preview changes without executing

    Returns:
        True if successful, False otherwise
    """
    # Current direct index name
    current_index = f"{index_prefix}_{env}_current"
    # Target versioned index name
    versioned_index = f"{index_prefix}_{env}_v{target_version}"
    # Alias name
    alias_name = current_index

    logger.info("=" * 80)
    logger.info("ES Alias Migration Strategy")
    logger.info("=" * 80)
    logger.info(f"ES Host: {es_host}")
    logger.info(f"Environment: {env}")
    logger.info(f"Current direct index: {current_index}")
    logger.info(f"Target versioned index: {versioned_index}")
    logger.info(f"Alias name: {alias_name}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 80)

    # Initialize ES client
    es_client = Elasticsearch([es_host], verify_certs=False)

    # Check cluster health
    try:
        health = es_client.cluster.health()
        logger.info(f"Cluster health: {health['status']}")
    except Exception as e:
        logger.error(f"Failed to connect to ES: {e}")
        return False

    # Step 1: Check if current index exists
    logger.info("\n[Step 1] Checking current index...")
    if not es_client.indices.exists(index=current_index):
        logger.error(f"Current index {current_index} does not exist!")
        return False

    # Get current index stats
    stats = es_client.indices.stats(index=current_index)
    doc_count = stats["indices"][current_index]["total"]["docs"]["count"]
    size_bytes = stats["indices"][current_index]["total"]["store"]["size_in_bytes"]
    size_gb = size_bytes / (1024**3)

    logger.info(f"  ✓ Current index exists: {current_index}")
    logger.info(f"  ✓ Documents: {doc_count:,}")
    logger.info(f"  ✓ Size: {size_gb:.2f} GB")

    # Get mapping to extract dims
    mapping_resp = es_client.indices.get_mapping(index=current_index)
    embedding_props = (
        mapping_resp.get(current_index, {})
        .get("mappings", {})
        .get("properties", {})
        .get("embedding", {})
    )
    current_dims = embedding_props.get("dims")
    logger.info(f"  ✓ Embedding dimensions: {current_dims}")

    # Step 2: Check if versioned index already exists
    logger.info(f"\n[Step 2] Checking versioned index {versioned_index}...")
    if es_client.indices.exists(index=versioned_index):
        logger.warning(f"  ⚠ Versioned index {versioned_index} already exists!")
        logger.warning("  This might indicate migration was already done or partially completed.")

        # Check if it's an alias pointing to it
        try:
            aliases = es_client.indices.get_alias(name=current_index)
            if versioned_index in aliases:
                logger.info(f"  ✓ Alias {alias_name} already points to {versioned_index}")
                logger.info("  Migration appears to be complete!")
                return True
        except NotFoundError:
            pass

        # Ask user what to do
        if not dry_run:
            response = input(
                f"Versioned index {versioned_index} exists. Delete and recreate? [y/N]: "
            )
            if response.lower() != "y":
                logger.info("Migration aborted by user")
                return False
    else:
        logger.info(f"  ✓ Versioned index {versioned_index} does not exist (will create)")

    # Step 3: Create versioned index (if not exists or user confirmed deletion)
    logger.info(f"\n[Step 3] Creating versioned index {versioned_index}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would create index: {versioned_index}")
        logger.info(f"  [DRY RUN] With dims: {current_dims}")
    else:
        manager = EsIndexManager(
            es_client=es_client,
            env=env,
            index_prefix=index_prefix,
        )

        try:
            manager.create_index(
                version=target_version,
                dims=current_dims or 768,
                skip_if_exists=False,
                validate_dims=False,  # Skip validation since we're migrating
            )
            logger.info(f"  ✓ Created versioned index: {versioned_index}")
        except Exception as e:
            logger.error(f"  ✗ Failed to create index: {e}")
            return False

    # Step 4: Reindex data
    logger.info(f"\n[Step 4] Reindexing data from {current_index} to {versioned_index}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would reindex {doc_count:,} documents")
    else:
        try:
            logger.info(f"  Reindexing {doc_count:,} documents (this may take a while)...")
            result = es_client.reindex(
                body={
                    "source": {"index": current_index},
                    "dest": {"index": versioned_index},
                },
                wait_for_completion=True,
                refresh=True,
            )

            created = result.get("created", 0)
            logger.info(f"  ✓ Reindexed {created:,} documents")

            if created != doc_count:
                logger.warning(
                    f"  ⚠ Document count mismatch: expected {doc_count:,}, got {created:,}"
                )
        except Exception as e:
            logger.error(f"  ✗ Reindex failed: {e}")
            return False

    # Step 5: Delete old direct index
    logger.info(f"\n[Step 5] Deleting old direct index {current_index}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would delete index: {current_index}")
    else:
        try:
            es_client.indices.delete(index=current_index)
            logger.info(f"  ✓ Deleted old index: {current_index}")
        except Exception as e:
            logger.error(f"  ✗ Failed to delete old index: {e}")
            logger.warning("  You may need to manually delete it later")

    # Step 6: Create alias
    logger.info(f"\n[Step 6] Creating alias {alias_name} → {versioned_index}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would create alias: {alias_name} → {versioned_index}")
    else:
        try:
            manager = EsIndexManager(
                es_client=es_client,
                env=env,
                index_prefix=index_prefix,
            )
            manager.switch_alias(version=target_version)
            logger.info(f"  ✓ Created alias: {alias_name} → {versioned_index}")
        except Exception as e:
            logger.error(f"  ✗ Failed to create alias: {e}")
            return False

    # Step 7: Verification
    logger.info("\n[Step 7] Verifying migration...")

    if dry_run:
        logger.info("  [DRY RUN] Verification skipped")
    else:
        try:
            # Check alias exists and points to correct index
            alias_resp = es_client.indices.get_alias(name=alias_name)
            if versioned_index in alias_resp:
                logger.info(f"  ✓ Alias verified: {alias_name} → {versioned_index}")
            else:
                logger.error(f"  ✗ Alias verification failed!")
                return False

            # Check document count
            new_stats = es_client.indices.stats(index=alias_name)
            new_doc_count = new_stats["indices"][versioned_index]["total"]["docs"]["count"]

            if new_doc_count == doc_count:
                logger.info(f"  ✓ Document count verified: {new_doc_count:,}")
            else:
                logger.warning(
                    f"  ⚠ Document count mismatch: expected {doc_count:,}, got {new_doc_count:,}"
                )

        except Exception as e:
            logger.error(f"  ✗ Verification failed: {e}")
            return False

    # Success!
    logger.info("\n" + "=" * 80)
    if dry_run:
        logger.info("✓ DRY RUN COMPLETE - No changes were made")
        logger.info("  Run without --dry-run to execute migration")
    else:
        logger.info("✓ MIGRATION COMPLETE")
        logger.info(f"  Old: {current_index} (direct index)")
        logger.info(f"  New: {versioned_index} (versioned index)")
        logger.info(f"  Alias: {alias_name} → {versioned_index}")
        logger.info("\n  Next steps:")
        logger.info("  1. Test search functionality with the alias")
        logger.info("  2. Verify ingestion works correctly")
        logger.info("  3. For future updates, create v2, v3, etc. and switch alias")
    logger.info("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ES index to alias-based naming strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--es-host",
        default=search_settings.es_host,
        help=f"Elasticsearch host (default: {search_settings.es_host})",
    )
    parser.add_argument(
        "--env",
        default=search_settings.es_env,
        help=f"Environment name (default: {search_settings.es_env})",
    )
    parser.add_argument(
        "--index-prefix",
        default=search_settings.es_index_prefix,
        help=f"Index prefix (default: {search_settings.es_index_prefix})",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Target version number (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )

    args = parser.parse_args()

    success = migrate_to_alias_strategy(
        es_host=args.es_host,
        env=args.env,
        index_prefix=args.index_prefix,
        target_version=args.version,
        dry_run=args.dry_run,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
