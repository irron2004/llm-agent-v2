#!/usr/bin/env python3
"""Validate embedding dimensions across all components.

This script checks dimension consistency across:
1. Configuration (.env, settings.py)
2. ES index mapping
3. ES stored documents
4. Embedder model

Usage:
    python scripts/validate_embedding_dimensions.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from elasticsearch import Elasticsearch

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.elasticsearch import EsIndexManager
from backend.services.embedding_service import EmbeddingService


def validate_dimensions() -> bool:
    """Validate embedding dimensions across all components.

    Returns:
        True if all dimensions match, False otherwise
    """
    print("=" * 80)
    print("Embedding Dimension Validation")
    print("=" * 80)
    print()

    results = {}
    all_match = True

    # 1. Configuration
    print("[1] Configuration")
    print("-" * 80)
    config_dims = search_settings.es_embedding_dims
    print(f"  SEARCH_ES_EMBEDDING_DIMS: {config_dims}")
    print(f"  RAG_EMBEDDING_METHOD: {rag_settings.embedding_method}")
    print(f"  RAG_EMBEDDING_VERSION: {rag_settings.embedding_version}")
    results["config"] = config_dims
    print()

    # 2. Embedder
    print("[2] Embedder Model")
    print("-" * 80)
    try:
        svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
        )
        embedder = svc.get_raw_embedder()

        # Get declared dimension
        embedder_dims = embedder.get_dimension()
        print(f"  Embedder.get_dimension(): {embedder_dims}")

        # Test actual embedding
        test_embedding = embedder.embed_batch(["test"])[0]
        actual_dims = len(test_embedding)
        print(f"  Actual generated dimension: {actual_dims}")

        results["embedder_declared"] = embedder_dims
        results["embedder_actual"] = actual_dims

        if embedder_dims != actual_dims:
            print(f"  ❌ Embedder dimension mismatch: declared={embedder_dims}, actual={actual_dims}")
            all_match = False
        else:
            print(f"  ✅ Embedder dimensions consistent")

    except Exception as e:
        print(f"  ❌ Error loading embedder: {e}")
        results["embedder_declared"] = None
        results["embedder_actual"] = None
        all_match = False

    print()

    # 3. ES Index Mapping
    print("[3] ES Index Mapping")
    print("-" * 80)
    try:
        es_client = Elasticsearch([search_settings.es_host], verify_certs=False)
        manager = EsIndexManager(
            es_client=es_client,
            env=search_settings.es_env,
            index_prefix=search_settings.es_index_prefix,
        )

        alias_name = manager.get_alias_name()
        print(f"  Alias: {alias_name}")

        # Check if alias exists
        try:
            es_dims = manager.get_index_dims(use_alias=True)
            if es_dims is not None:
                print(f"  Mapping dims: {es_dims}")
                results["es_mapping"] = es_dims
            else:
                print(f"  ⚠️  Could not read mapping dims from {alias_name}")
                results["es_mapping"] = None
        except Exception as e:
            print(f"  ⚠️  Index/alias not found: {e}")
            results["es_mapping"] = None

    except Exception as e:
        print(f"  ❌ Error connecting to ES: {e}")
        results["es_mapping"] = None
        all_match = False

    print()

    # 4. ES Stored Documents
    print("[4] ES Stored Documents")
    print("-" * 80)
    try:
        resp = es_client.search(
            index=alias_name,
            body={"query": {"match_all": {}}, "size": 1},
        )

        if resp["hits"]["hits"]:
            hit = resp["hits"]["hits"][0]
            embedding = hit["_source"].get("embedding", [])
            stored_dims = len(embedding)
            print(f"  Sample document embedding dimension: {stored_dims}")
            print(f"  First 5 values: {embedding[:5]}")
            results["es_stored"] = stored_dims
        else:
            print(f"  ⚠️  No documents found in {alias_name}")
            results["es_stored"] = None

    except Exception as e:
        print(f"  ❌ Error reading documents: {e}")
        results["es_stored"] = None
        all_match = False

    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Print all dimensions
    for key, value in results.items():
        status = "✅" if value == config_dims else "❌"
        print(f"  {status} {key:20s}: {value}")

    print()

    # Check consistency
    non_none_dims = [v for v in results.values() if v is not None]
    if len(set(non_none_dims)) == 1 and non_none_dims[0] == config_dims:
        print("✅ ALL DIMENSIONS MATCH!")
        print(f"   Consistent dimension: {config_dims}")
        return True
    else:
        print("❌ DIMENSION MISMATCH DETECTED!")
        print()
        print("Recommended actions:")
        print("  1. Check .env file: SEARCH_ES_EMBEDDING_DIMS")
        print("  2. Check embedding model: RAG_EMBEDDING_METHOD")
        print("  3. If changing dims, create new ES index and reindex")
        print()
        return False


if __name__ == "__main__":
    success = validate_dimensions()
    sys.exit(0 if success else 1)
