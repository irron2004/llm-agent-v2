#!/usr/bin/env python3
"""Validate embedding dimension consistency across all components.

Checks:
1. .env SEARCH_ES_EMBEDDING_DIMS
2. Actual embedder dimension
3. Actual ES index mapping dimension
4. Reports any mismatches
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.parent))

import requests
from backend.config.settings import search_settings, rag_settings
from backend.llm_infrastructure.embedding import get_embedder


def get_es_index_dims(es_host: str, index_name: str) -> int | None:
    """Get embedding dimension from ES index mapping."""
    try:
        resp = requests.get(f"{es_host}/{index_name}/_mapping", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Navigate to embedding field
        for idx_name, idx_data in data.items():
            props = idx_data.get("mappings", {}).get("properties", {})
            embedding = props.get("embedding", {})
            dims = embedding.get("dims")
            if dims is not None:
                return int(dims)
        return None
    except Exception as e:
        print(f"⚠️  Failed to get ES index dims: {e}")
        return None


def main():
    print("=" * 70)
    print("🔍 Embedding Dimension Configuration Validation")
    print("=" * 70)
    print()

    # 1. Environment variable
    env_dims = os.getenv("SEARCH_ES_EMBEDDING_DIMS")
    print(f"1️⃣  Environment Variable (SEARCH_ES_EMBEDDING_DIMS)")
    print(f"   Value: {env_dims}")
    print()

    # 2. Pydantic settings (after env override)
    config_dims = search_settings.es_embedding_dims
    print(f"2️⃣  Pydantic Settings (search_settings.es_embedding_dims)")
    print(f"   Value: {config_dims}")
    print(f"   Source: {'env override' if env_dims else 'code default'}")
    print()

    # 3. RAG embedding method
    embedding_method = rag_settings.embedding_method
    embedding_version = rag_settings.embedding_version
    print(f"3️⃣  RAG Embedding Method")
    print(f"   Method: {embedding_method}")
    print(f"   Version: {embedding_version}")

    try:
        embedder = get_embedder(
            embedding_method,
            version=embedding_version,
            device=rag_settings.embedding_device,
            normalize_embeddings=rag_settings.vector_normalize,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
        embedder_dims = embedder.get_dimension()
        print(f"   Actual Dimension: {embedder_dims}")
    except Exception as e:
        print(f"   ⚠️  Failed to initialize embedder: {e}")
        embedder_dims = None
    print()

    # 4. Elasticsearch index
    es_host = search_settings.es_host
    es_env = search_settings.es_env
    es_prefix = search_settings.es_index_prefix

    # Try common index patterns
    index_candidates = [
        f"{es_prefix}_{es_env}_current",
        f"{es_prefix}_{es_env}",
        f"{es_prefix}",
    ]

    print(f"4️⃣  Elasticsearch Index")
    print(f"   ES Host: {es_host}")
    print(f"   Index Prefix: {es_prefix}")
    print(f"   Environment: {es_env}")

    es_dims = None
    found_index = None
    for idx_name in index_candidates:
        dims = get_es_index_dims(es_host, idx_name)
        if dims is not None:
            es_dims = dims
            found_index = idx_name
            break

    if es_dims:
        print(f"   Found Index: {found_index}")
        print(f"   Embedding Dimension: {es_dims}")
    else:
        print(f"   ⚠️  No index found (tried: {', '.join(index_candidates)})")
    print()

    # 5. Validation summary
    print("=" * 70)
    print("📊 Validation Summary")
    print("=" * 70)
    print()

    all_dims = {
        "Config (settings)": config_dims,
        "Embedder (runtime)": embedder_dims,
        "ES Index (actual)": es_dims,
    }

    print("Dimensions:")
    for name, value in all_dims.items():
        status = "✅" if value is not None else "⚠️ "
        print(f"  {status} {name:25s}: {value if value is not None else 'N/A'}")
    print()

    # Check consistency
    valid_dims = [v for v in all_dims.values() if v is not None]
    if len(valid_dims) < 2:
        print("⚠️  Not enough data to validate consistency")
        return 0

    if len(set(valid_dims)) == 1:
        print("✅ All dimensions are consistent!")
        return 0
    else:
        print("❌ DIMENSION MISMATCH DETECTED!")
        print()
        print("Recommended actions:")
        print("  1. Ensure SEARCH_ES_EMBEDDING_DIMS matches your embedder")
        print("  2. Ensure ES index mapping matches your embedder")
        print("  3. Consider reindexing if embedder was changed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
