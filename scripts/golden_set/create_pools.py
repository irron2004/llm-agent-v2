"""Golden Set Pooling Script.

Creates candidate document pools for each test question using multiple
search methods (BM25, Dense, Hybrid, Stratified).

Usage:
    # Basic usage
    python scripts/golden_set/create_pools.py

    # With options
    python scripts/golden_set/create_pools.py \
        --queries data/golden_set/queries.jsonl \
        --output data/golden_set/pools/ \
        --pool-size 150 \
        --top-k 50 \
        --min-per-type 10

    # Using specific ES host
    python scripts/golden_set/create_pools.py --es-host http://localhost:9200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticsearch import Elasticsearch
from tqdm import tqdm

from backend.config.settings import search_settings, rag_settings
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.embedding import get_embedder

from scripts.golden_set.config import PoolingConfig, QueryPool
from scripts.golden_set.pooling import (
    BM25Pooler,
    DensePooler,
    HybridPooler,
    StratifiedPooler,
)
from scripts.golden_set.deduplication import merge_and_deduplicate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_es_client(host: str | None = None) -> Elasticsearch:
    """Create Elasticsearch client."""
    es_host = host or search_settings.es_host
    client_kwargs: dict[str, Any] = {"hosts": [es_host], "verify_certs": True}

    if search_settings.es_user and search_settings.es_password:
        client_kwargs["basic_auth"] = (
            search_settings.es_user,
            search_settings.es_password,
        )

    return Elasticsearch(**client_kwargs)


def get_index_name() -> str:
    """Get current index alias name."""
    return f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"


def load_queries(queries_path: Path) -> list[dict[str, Any]]:
    """Load queries from JSONL or JSON file."""
    if queries_path.suffix == ".jsonl":
        queries = []
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        return queries
    elif queries_path.suffix == ".json":
        with open(queries_path, encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {queries_path.suffix}")


def create_pool_for_query(
    query: dict[str, Any],
    poolers: list,
    config: PoolingConfig,
) -> QueryPool:
    """Create pool for a single query."""
    query_id = query["id"]
    query_text = query["question"]
    category = query.get("category", "unknown")
    difficulty = query.get("difficulty", "medium")

    # Collect from each pooler
    all_pools = []
    method_counts: dict[str, int] = {}

    for pooler in poolers:
        docs = pooler.pool(query_text, config.top_k_per_method)
        all_pools.append(docs)
        method_counts[pooler.method_name] = len(docs)

    # Merge and deduplicate
    final_docs = merge_and_deduplicate(all_pools, config)

    # Compute statistics
    doc_type_counts: dict[str, int] = {}
    source_method_counts: dict[str, int] = {}

    for doc in final_docs:
        doc_type_counts[doc.doc_type] = doc_type_counts.get(doc.doc_type, 0) + 1
        source_method_counts[doc.source_method] = (
            source_method_counts.get(doc.source_method, 0) + 1
        )

    pool_stats = {
        "total_before_dedup": sum(len(p) for p in all_pools),
        "total_after_dedup": len(final_docs),
        "by_method_before": method_counts,
        "by_method_after": source_method_counts,
        "by_doc_type": doc_type_counts,
    }

    return QueryPool(
        query_id=query_id,
        query_text=query_text,
        category=category,
        difficulty=difficulty,
        documents=final_docs,
        pool_stats=pool_stats,
    )


def save_pool(pool: QueryPool, output_dir: Path) -> None:
    """Save pool to JSONL file."""
    output_path = output_dir / f"{pool.query_id}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        # First line: metadata
        meta = {
            "_meta": True,
            "query_id": pool.query_id,
            "query_text": pool.query_text,
            "category": pool.category,
            "difficulty": pool.difficulty,
            "pool_stats": pool.pool_stats,
            "created_at": datetime.now().isoformat(),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Remaining lines: documents
        for doc in pool.documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden Set Pooling")
    parser.add_argument(
        "--queries",
        type=str,
        default="data/golden_set/queries.jsonl",
        help="Input queries file (JSONL or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/golden_set/pools/",
        help="Output directory for pools",
    )
    parser.add_argument(
        "--es-host",
        type=str,
        default=None,
        help="Elasticsearch host (default: from settings)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=150,
        help="Maximum pool size per query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k per search method",
    )
    parser.add_argument(
        "--min-per-type",
        type=int,
        default=10,
        help="Minimum documents per doc_type",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Near-duplicate similarity threshold",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        help="Embedder name (default: from rag_settings)",
    )

    args = parser.parse_args()

    # Paths
    queries_path = ROOT / args.queries
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Elasticsearch connection
    logger.info("Connecting to Elasticsearch...")
    es = get_es_client(args.es_host)
    if not es.ping():
        logger.error("Cannot connect to Elasticsearch")
        sys.exit(1)

    index_name = get_index_name()
    logger.info(f"Using index: {index_name}")

    # ES engine
    es_engine = EsSearchEngine(
        es_client=es,
        index_name=index_name,
        text_fields=["search_text", "chunk_summary^0.7"],
    )

    # Configuration
    config = PoolingConfig(
        top_k_per_method=args.top_k,
        total_pool_size=args.pool_size,
        min_per_doc_type=args.min_per_type,
        similarity_threshold=args.similarity_threshold,
    )

    # Embedder
    embedder_name = args.embedder or rag_settings.embedding_method
    logger.info(f"Loading embedder: {embedder_name}")
    embedder = get_embedder(embedder_name)

    # Initialize poolers
    logger.info("Initializing poolers...")
    poolers = [
        BM25Pooler(es_engine, config),
        DensePooler(es_engine, config, embedder),
        HybridPooler(es_engine, config, embedder),
        StratifiedPooler(es_engine, config, embedder),
    ]

    # Load queries
    logger.info(f"Loading queries from {queries_path}...")
    queries = load_queries(queries_path)
    logger.info(f"Found {len(queries)} queries")

    # Create pools
    logger.info("\nCreating pools...")
    all_stats = []

    for query in tqdm(queries, desc="Pooling"):
        pool = create_pool_for_query(query, poolers, config)
        save_pool(pool, output_dir)
        all_stats.append(
            {
                "query_id": pool.query_id,
                "query_text": pool.query_text[:50] + "...",
                **pool.pool_stats,
            }
        )

    # Save overall statistics
    stats_path = output_dir.parent / "pool_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "config": {
                    "top_k_per_method": config.top_k_per_method,
                    "total_pool_size": config.total_pool_size,
                    "min_per_doc_type": config.min_per_doc_type,
                    "similarity_threshold": config.similarity_threshold,
                    "embedder": embedder_name,
                    "index": index_name,
                },
                "queries": all_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Summary
    logger.info(f"\nDone! Pools saved to {output_dir}")
    logger.info(f"Statistics saved to {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Pool Creation Summary")
    print("=" * 60)
    for stat in all_stats:
        print(f"\n{stat['query_id']}: {stat['query_text']}")
        print(f"  Before dedup: {stat['total_before_dedup']}")
        print(f"  After dedup:  {stat['total_after_dedup']}")
        print(f"  By doc_type:  {stat['by_doc_type']}")


if __name__ == "__main__":
    main()
