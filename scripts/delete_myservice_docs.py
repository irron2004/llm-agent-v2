"""Delete all myservice documents from ES.

Usage:
    python scripts/delete_myservice_docs.py
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from elasticsearch import Elasticsearch
from backend.config.settings import search_settings


def main():
    """Delete all myservice documents."""
    # Connect to ES
    es = Elasticsearch(hosts=[search_settings.es_host])
    index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

    print("=" * 80)
    print("DELETE MYSERVICE DOCUMENTS")
    print("=" * 80)

    # Check current count
    try:
        count_result = es.count(
            index=index,
            body={"query": {"term": {"doc_type.keyword": "myservice"}}}
        )
        current_count = count_result["count"]
    except Exception as e:
        print(f"Error counting documents: {e}")
        sys.exit(1)

    print(f"Current myservice documents: {current_count}")

    if current_count == 0:
        print("\nNo myservice documents to delete.")
        print("=" * 80)
        return

    # Confirm deletion
    print("\n⚠️  WARNING: This will delete ALL myservice documents!")
    response = input(f"Delete {current_count} documents? (yes/no): ")

    if response.lower() != "yes":
        print("\nDeletion cancelled.")
        print("=" * 80)
        return

    # Delete by query
    print("\nDeleting documents...")
    try:
        delete_result = es.delete_by_query(
            index=index,
            body={"query": {"term": {"doc_type.keyword": "myservice"}}},
            refresh=True
        )

        deleted = delete_result.get("deleted", 0)
        print(f"✓ Deleted {deleted} documents")

    except Exception as e:
        print(f"✗ Error deleting documents: {e}")
        sys.exit(1)

    # Verify deletion
    try:
        verify_result = es.count(
            index=index,
            body={"query": {"term": {"doc_type.keyword": "myservice"}}}
        )
        remaining = verify_result["count"]
        print(f"Remaining myservice documents: {remaining}")
    except Exception:
        pass

    print("=" * 80)
    print("✓ Deletion complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
