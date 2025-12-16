"""Vector database abstraction layer.

Provides a unified interface for vector database operations,
allowing easy switching between different backends (ES, Pinecone, Milvus, etc.).

Usage:
    from backend.llm_infrastructure.vectordb import get_vectordb

    # Get ES backend
    db = get_vectordb("elasticsearch", es_client=client, index_name="my_index")

    # Upsert documents
    result = db.upsert([{"id": "1", "embedding": [...], "content": "..."}])

    # Search
    hits = db.search(query_vector=[...], top_k=10)
"""

from backend.llm_infrastructure.vectordb.base import (
    VectorDBClient,
    SearchHit,
    UpsertResult,
    DeleteResult,
)
from backend.llm_infrastructure.vectordb.registry import (
    VectorDBRegistry,
    get_vectordb,
    register_vectordb,
)

__all__ = [
    "VectorDBClient",
    "SearchHit",
    "UpsertResult",
    "DeleteResult",
    "VectorDBRegistry",
    "get_vectordb",
    "register_vectordb",
]
