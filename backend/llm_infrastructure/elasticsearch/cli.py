#!/usr/bin/env python3
"""CLI for Elasticsearch index management.

Usage:
    # Create index
    python -m backend.llm_infrastructure.elasticsearch.cli create --version 1 --dims 1024

    # Delete index
    python -m backend.llm_infrastructure.elasticsearch.cli delete --version 1

    # Switch alias
    python -m backend.llm_infrastructure.elasticsearch.cli switch --version 1

    # List indices
    python -m backend.llm_infrastructure.elasticsearch.cli list

    # Show index info
    python -m backend.llm_infrastructure.elasticsearch.cli info --version 1

    # Health check
    python -m backend.llm_infrastructure.elasticsearch.cli health
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .manager import EsIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_manager_from_env() -> EsIndexManager:
    """Create EsIndexManager from environment variables."""
    return EsIndexManager(
        es_host=os.getenv("ES_HOST", "http://localhost:9200"),
        env=os.getenv("ES_ENV", "dev"),
        index_prefix=os.getenv("ES_INDEX_PREFIX", "rag_chunks"),
        es_user=os.getenv("ES_USER") or None,
        es_password=os.getenv("ES_PASSWORD") or None,
    )


def cmd_create(args: argparse.Namespace) -> int:
    """Create a new index."""
    manager = get_manager_from_env()

    try:
        result = manager.create_index(
            version=args.version,
            dims=args.dims,
            number_of_shards=args.shards,
            number_of_replicas=args.replicas,
            embedding_model=args.embedding_model,
            chunking_method=args.chunking_method,
            chunking_size=args.chunking_size,
            chunking_overlap=args.chunking_overlap,
            preprocess_method=args.preprocess_method,
            skip_if_exists=args.skip_if_exists,
        )
        print(f"Index created: {manager.get_index_name(args.version)}")
        if result.get("skipped"):
            print("  (already existed, skipped)")

        if args.switch_alias:
            manager.switch_alias(args.version)
            print(f"Alias switched: {manager.get_alias_name()} -> {manager.get_index_name(args.version)}")

        return 0
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        return 1


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete an index."""
    manager = get_manager_from_env()
    index_name = manager.get_index_name(args.version)

    if not args.force:
        confirm = input(f"Are you sure you want to delete {index_name}? [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            return 0

    try:
        result = manager.delete_index(version=args.version)
        if result.get("not_found"):
            print(f"Index {index_name} not found")
        else:
            print(f"Index deleted: {index_name}")
        return 0
    except Exception as e:
        logger.error(f"Failed to delete index: {e}")
        return 1


def cmd_switch(args: argparse.Namespace) -> int:
    """Switch alias to a specific version."""
    manager = get_manager_from_env()

    try:
        current = manager.get_alias_target()
        manager.switch_alias(args.version)
        new_index = manager.get_index_name(args.version)
        print(f"Alias switched: {manager.get_alias_name()}")
        if current:
            print(f"  From: {current}")
        print(f"  To:   {new_index}")
        return 0
    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Failed to switch alias: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all indices."""
    manager = get_manager_from_env()

    try:
        indices = manager.list_indices()
        alias_target = manager.get_alias_target()
        alias_name = manager.get_alias_name()

        print(f"Environment: {manager.env}")
        print(f"Alias: {alias_name} -> {alias_target or '(not set)'}")
        print(f"\nIndices ({len(indices)}):")

        if not indices:
            print("  (none)")
        else:
            for index in sorted(indices):
                marker = " *" if index == alias_target else ""
                print(f"  - {index}{marker}")

        return 0
    except Exception as e:
        logger.error(f"Failed to list indices: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show index information."""
    manager = get_manager_from_env()
    index_name = manager.get_index_name(args.version)

    try:
        info = manager.get_index_info(args.version)
        if not info:
            print(f"Index {index_name} not found")
            return 1

        print(f"Index: {index_name}")
        print(f"\nSettings:")
        print(json.dumps(info.get("settings", {}), indent=2))
        print(f"\nMappings _meta:")
        meta = info.get("mappings", {}).get("_meta", {})
        print(json.dumps(meta, indent=2))

        if args.full:
            print(f"\nFull Mappings:")
            print(json.dumps(info.get("mappings", {}), indent=2))

        return 0
    except Exception as e:
        logger.error(f"Failed to get index info: {e}")
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Check Elasticsearch health."""
    manager = get_manager_from_env()

    try:
        health = manager.health_check()
        status = health.get("status", "unknown")
        status_icon = {"green": "✓", "yellow": "⚠", "red": "✗"}.get(status, "?")

        print(f"Cluster: {health.get('cluster_name', 'unknown')}")
        print(f"Status:  {status_icon} {status}")
        print(f"Nodes:   {health.get('number_of_nodes', 0)}")
        print(f"Shards:  {health.get('active_shards', 0)} active")
        return 0
    except Exception as e:
        logger.error(f"Failed to check health: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Elasticsearch index management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  ES_HOST          Elasticsearch host (default: http://localhost:9200)
  ES_ENV           Environment name (default: dev)
  ES_INDEX_PREFIX  Index prefix (default: rag_chunks)
  ES_USER          Elasticsearch username (optional)
  ES_PASSWORD      Elasticsearch password (optional)

Examples:
  # Create dev index v1 with 1024 dims and switch alias
  %(prog)s create --version 1 --dims 1024 --switch-alias

  # List all indices
  %(prog)s list

  # Switch alias to v2
  %(prog)s switch --version 2

  # Delete old index
  %(prog)s delete --version 1 --force
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = subparsers.add_parser("create", help="Create a new index")
    p_create.add_argument("--version", "-v", type=int, required=True, help="Index version number")
    p_create.add_argument("--dims", "-d", type=int, default=1024, help="Embedding dimensions (default: 1024)")
    p_create.add_argument("--shards", type=int, default=1, help="Number of shards (default: 1)")
    p_create.add_argument("--replicas", type=int, default=0, help="Number of replicas (default: 0)")
    p_create.add_argument("--embedding-model", default="nlpai-lab/KoE5", help="Embedding model name")
    p_create.add_argument("--chunking-method", default="fixed_size", help="Chunking method")
    p_create.add_argument("--chunking-size", type=int, default=512, help="Chunk size")
    p_create.add_argument("--chunking-overlap", type=int, default=50, help="Chunk overlap")
    p_create.add_argument("--preprocess-method", default="normalize", help="Preprocessing method")
    p_create.add_argument("--skip-if-exists", action="store_true", help="Skip if index already exists")
    p_create.add_argument("--switch-alias", "-s", action="store_true", help="Switch alias after creation")
    p_create.set_defaults(func=cmd_create)

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete an index")
    p_delete.add_argument("--version", "-v", type=int, required=True, help="Index version number")
    p_delete.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    p_delete.set_defaults(func=cmd_delete)

    # switch
    p_switch = subparsers.add_parser("switch", help="Switch alias to an index version")
    p_switch.add_argument("--version", "-v", type=int, required=True, help="Target index version")
    p_switch.set_defaults(func=cmd_switch)

    # list
    p_list = subparsers.add_parser("list", help="List all indices")
    p_list.set_defaults(func=cmd_list)

    # info
    p_info = subparsers.add_parser("info", help="Show index information")
    p_info.add_argument("--version", "-v", type=int, required=True, help="Index version number")
    p_info.add_argument("--full", "-f", action="store_true", help="Show full mappings")
    p_info.set_defaults(func=cmd_info)

    # health
    p_health = subparsers.add_parser("health", help="Check Elasticsearch health")
    p_health.set_defaults(func=cmd_health)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
