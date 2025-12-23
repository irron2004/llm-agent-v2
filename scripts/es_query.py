"""Elasticsearch query utility for inspecting indexed documents.

Usage:
    # Index statistics
    python scripts/es_query.py stats

    # List all unique doc_ids
    python scripts/es_query.py list-docs

    # Get chunks for a specific doc_id
    python scripts/es_query.py get-doc --doc-id "my_document_id"

    # Search documents
    python scripts/es_query.py search --query "검색어"

    # All commands support --host option for non-Docker execution
    python scripts/es_query.py stats --host http://localhost:8002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticsearch import Elasticsearch

from backend.config.settings import search_settings


def get_es_client(host: str | None = None) -> Elasticsearch:
    """Create Elasticsearch client."""
    es_host = host or search_settings.es_host
    client_kwargs: dict[str, Any] = {
        "hosts": [es_host],
        "verify_certs": True,
    }
    if search_settings.es_user and search_settings.es_password:
        client_kwargs["basic_auth"] = (
            search_settings.es_user,
            search_settings.es_password,
        )
    return Elasticsearch(**client_kwargs)


def get_index_name() -> str:
    """Get current index alias name."""
    return f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"


def cmd_stats(es: Elasticsearch, index: str) -> None:
    """Show index statistics."""
    print("=" * 60)
    print("Index Statistics")
    print("=" * 60)
    print(f"Index: {index}")
    print()

    # Check if index exists
    if not es.indices.exists(index=index):
        print(f"ERROR: Index '{index}' does not exist")
        return

    # Get document count
    count_result = es.count(index=index)
    total_docs = count_result.get("count", 0)
    print(f"Total chunks: {total_docs:,}")

    # Get unique doc_ids count
    # Use .keyword suffix for aggregation (text fields don't support aggregation)
    agg_result = es.search(
        index=index,
        body={
            "size": 0,
            "aggs": {
                "unique_docs": {"cardinality": {"field": "doc_id.keyword"}},
                "doc_types": {"terms": {"field": "doc_type.keyword", "size": 20}},
                "languages": {"terms": {"field": "lang.keyword", "size": 10}},
            },
        },
    )

    aggs = agg_result.get("aggregations", {})
    unique_docs = aggs.get("unique_docs", {}).get("value", 0)
    print(f"Unique documents: {unique_docs:,}")
    print()

    # Doc types
    doc_types = aggs.get("doc_types", {}).get("buckets", [])
    if doc_types:
        print("Document types:")
        for bucket in doc_types:
            print(f"  - {bucket['key']}: {bucket['doc_count']:,}")
        print()

    # Languages
    languages = aggs.get("languages", {}).get("buckets", [])
    if languages:
        print("Languages:")
        for bucket in languages:
            print(f"  - {bucket['key']}: {bucket['doc_count']:,}")
        print()

    # Index settings
    settings = es.indices.get_settings(index=index)
    # Resolve alias to actual index
    actual_index = list(settings.keys())[0] if settings else index
    index_settings = settings.get(actual_index, {}).get("settings", {}).get("index", {})
    print(f"Number of shards: {index_settings.get('number_of_shards', 'N/A')}")
    print(f"Number of replicas: {index_settings.get('number_of_replicas', 'N/A')}")


def cmd_list_docs(es: Elasticsearch, index: str, limit: int = 100) -> None:
    """List all unique doc_ids."""
    print("=" * 60)
    print("Document List")
    print("=" * 60)
    print(f"Index: {index}")
    print()

    if not es.indices.exists(index=index):
        print(f"ERROR: Index '{index}' does not exist")
        return

    # Get unique doc_ids with chunk counts
    # Use .keyword suffix for aggregation (text fields don't support aggregation)
    result = es.search(
        index=index,
        body={
            "size": 0,
            "aggs": {
                "docs": {
                    "terms": {"field": "doc_id.keyword", "size": limit},
                    "aggs": {
                        "doc_type": {"terms": {"field": "doc_type.keyword", "size": 1}},
                        "page_range": {
                            "stats": {"field": "page"},
                        },
                    },
                }
            },
        },
    )

    buckets = result.get("aggregations", {}).get("docs", {}).get("buckets", [])

    if not buckets:
        print("No documents found")
        return

    print(f"Found {len(buckets)} documents:\n")
    print(f"{'doc_id':<60} {'chunks':>8} {'pages':>10} {'type':<15}")
    print("-" * 95)

    for bucket in buckets:
        doc_id = bucket["key"]
        chunk_count = bucket["doc_count"]
        page_stats = bucket.get("page_range", {})
        page_min = int(page_stats.get("min", 0))
        page_max = int(page_stats.get("max", 0))
        page_range = f"{page_min}-{page_max}" if page_min != page_max else str(page_min)

        doc_type_buckets = bucket.get("doc_type", {}).get("buckets", [])
        doc_type = doc_type_buckets[0]["key"] if doc_type_buckets else "N/A"

        # Truncate long doc_id
        display_id = doc_id[:57] + "..." if len(doc_id) > 60 else doc_id
        print(f"{display_id:<60} {chunk_count:>8} {page_range:>10} {doc_type:<15}")


def cmd_get_doc(es: Elasticsearch, index: str, doc_id: str, show_content: bool = True) -> None:
    """Get all chunks for a specific doc_id."""
    print("=" * 60)
    print(f"Document: {doc_id}")
    print("=" * 60)
    print()

    if not es.indices.exists(index=index):
        print(f"ERROR: Index '{index}' does not exist")
        return

    # Search for all chunks with this doc_id
    # Use doc_id.keyword for exact match on text field
    result = es.search(
        index=index,
        body={
            "query": {"term": {"doc_id.keyword": doc_id}},
            "size": 1000,
            "sort": [{"page": "asc"}, {"chunk_id": "asc"}],
            "_source": {
                "excludes": ["embedding"],  # Exclude large embedding vectors
            },
        },
    )

    hits = result.get("hits", {}).get("hits", [])

    if not hits:
        print(f"No chunks found for doc_id: {doc_id}")
        return

    print(f"Found {len(hits)} chunks:\n")

    for hit in hits:
        source = hit["_source"]
        chunk_id = source.get("chunk_id", "N/A")
        page = source.get("page", "N/A")
        doc_type = source.get("doc_type", "N/A")
        lang = source.get("lang", "N/A")
        content = source.get("content", "")
        created_at = source.get("created_at", "N/A")

        print(f"[Chunk: {chunk_id}]")
        print(f"  Page: {page} | Type: {doc_type} | Lang: {lang}")
        print(f"  Created: {created_at}")

        # Additional metadata
        device_name = source.get("device_name")
        chapter = source.get("chapter")
        if device_name:
            print(f"  Device: {device_name}")
        if chapter:
            print(f"  Chapter: {chapter}")

        if show_content:
            # Show truncated content
            display_content = content[:500] + "..." if len(content) > 500 else content
            print(f"  Content: {display_content}")
        print()


def cmd_search(
    es: Elasticsearch,
    index: str,
    query: str,
    top_k: int = 10,
    doc_type: str | None = None,
) -> None:
    """Search documents using BM25."""
    print("=" * 60)
    print(f"Search: {query}")
    print("=" * 60)
    print()

    if not es.indices.exists(index=index):
        print(f"ERROR: Index '{index}' does not exist")
        return

    # Build query
    must_clauses = [
        {
            "multi_match": {
                "query": query,
                "fields": ["content^2", "search_text", "doc_description", "chunk_summary"],
                "type": "best_fields",
            }
        }
    ]

    filter_clauses = []
    if doc_type:
        filter_clauses.append({"term": {"doc_type.keyword": doc_type}})

    search_body: dict[str, Any] = {
        "query": {
            "bool": {
                "must": must_clauses,
            }
        },
        "size": top_k,
        "_source": {"excludes": ["embedding"]},
        "highlight": {
            "fields": {
                "content": {"fragment_size": 150, "number_of_fragments": 2},
            }
        },
    }

    if filter_clauses:
        search_body["query"]["bool"]["filter"] = filter_clauses

    result = es.search(index=index, body=search_body)

    hits = result.get("hits", {}).get("hits", [])
    total = result.get("hits", {}).get("total", {}).get("value", 0)

    print(f"Total matches: {total} (showing top {len(hits)})\n")

    if not hits:
        print("No results found")
        return

    for i, hit in enumerate(hits, 1):
        source = hit["_source"]
        score = hit["_score"]
        doc_id = source.get("doc_id", "N/A")
        chunk_id = source.get("chunk_id", "N/A")
        page = source.get("page", "N/A")
        content = source.get("content", "")

        # Get highlights if available
        highlights = hit.get("highlight", {}).get("content", [])
        highlight_text = " ... ".join(highlights) if highlights else content[:200]

        print(f"{i}. [Score: {score:.4f}] {doc_id}")
        print(f"   Chunk: {chunk_id} | Page: {page}")
        print(f"   {highlight_text}")
        print()


def cmd_raw_query(es: Elasticsearch, index: str, query_file: str) -> None:
    """Execute a raw ES query from a JSON file."""
    print("=" * 60)
    print("Raw Query Execution")
    print("=" * 60)
    print()

    query_path = Path(query_file)
    if not query_path.exists():
        print(f"ERROR: Query file not found: {query_file}")
        return

    with open(query_path) as f:
        query_body = json.load(f)

    print(f"Query: {json.dumps(query_body, indent=2)[:500]}...")
    print()

    result = es.search(index=index, body=query_body)
    print(json.dumps(result.body, indent=2, ensure_ascii=False))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser."""
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Elasticsearch host (default: from settings, use localhost:8002 for local)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Index name (default: current alias)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ES document query utility")
    add_common_args(parser)

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    add_common_args(stats_parser)

    # list-docs command
    list_parser = subparsers.add_parser("list-docs", help="List all documents")
    add_common_args(list_parser)
    list_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of documents to list",
    )

    # get-doc command
    get_parser = subparsers.add_parser("get-doc", help="Get document chunks")
    add_common_args(get_parser)
    get_parser.add_argument(
        "--doc-id",
        type=str,
        required=True,
        help="Document ID to retrieve",
    )
    get_parser.add_argument(
        "--no-content",
        action="store_true",
        help="Hide chunk content",
    )

    # search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    add_common_args(search_parser)
    search_parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Search query",
    )
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results",
    )
    search_parser.add_argument(
        "--doc-type",
        type=str,
        default=None,
        help="Filter by document type",
    )

    # raw command
    raw_parser = subparsers.add_parser("raw", help="Execute raw query from JSON file")
    add_common_args(raw_parser)
    raw_parser.add_argument(
        "--query-file",
        type=str,
        required=True,
        help="Path to JSON query file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.command:
        print("ERROR: No command specified. Use --help for usage.")
        sys.exit(1)

    es = get_es_client(args.host)
    index = args.index or get_index_name()

    # Check connection
    if not es.ping():
        print(f"ERROR: Cannot connect to Elasticsearch")
        print(f"Host: {args.host or search_settings.es_host}")
        print()
        print("If running outside Docker, use: --host http://localhost:8002")
        sys.exit(1)

    if args.command == "stats":
        cmd_stats(es, index)
    elif args.command == "list-docs":
        cmd_list_docs(es, index, limit=args.limit)
    elif args.command == "get-doc":
        cmd_get_doc(es, index, args.doc_id, show_content=not args.no_content)
    elif args.command == "search":
        cmd_search(es, index, args.query, top_k=args.top_k, doc_type=args.doc_type)
    elif args.command == "raw":
        cmd_raw_query(es, index, args.query_file)


if __name__ == "__main__":
    main()
