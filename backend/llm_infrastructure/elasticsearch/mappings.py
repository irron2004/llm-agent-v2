"""Elasticsearch index mappings for RAG chunks.

Index naming convention:
    - Index: rag_chunks_{env}_v{version}  (e.g., rag_chunks_dev_v1)
    - Alias: rag_chunks_{env}_current     (e.g., rag_chunks_dev_current)

Rolling strategy:
    - Create new version index (v2, v3, ...)
    - Switch alias to new index
    - Old index can be deleted or kept for rollback
"""

from typing import Any


def get_rag_chunks_mapping(dims: int = 768) -> dict[str, Any]:
    """Get RAG chunks index mapping with specified embedding dimensions.

    Args:
        dims: Embedding vector dimensions (default: 768 for BGE-base, 1024 for KoE5/multilingual-e5)

    Returns:
        Elasticsearch mapping definition
    """
    return {
        "properties": {
            # ===================================================================
            # Primary Keys / Location
            # ===================================================================
            "doc_id": {
                "type": "keyword",
                "doc_values": True,
            },
            "chunk_id": {
                "type": "keyword",
                "doc_values": True,
            },
            "page": {
                "type": "integer",
            },
            # ===================================================================
            # Text Fields
            # ===================================================================
            "content": {
                "type": "text",
                "analyzer": "nori",  # Korean morphological analyzer
            },
            "search_text": {
                "type": "text",
                "analyzer": "nori",  # Korean morphological analyzer for BM25 search
                # Combined field: content + summary + caption + tags
                # Used for BM25 keyword search
            },
            # ===================================================================
            # Vector Embedding (for dense retrieval)
            # ===================================================================
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
                # ES 8.x+ supports HNSW by default
                # For custom HNSW params:
                # "index_options": {
                #     "type": "hnsw",
                #     "m": 16,
                #     "ef_construction": 100,
                # },
            },
            # ===================================================================
            # Metadata / Filter Fields
            # ===================================================================
            "lang": {
                "type": "keyword",
                "doc_values": True,
            },
            "doc_type": {
                "type": "keyword",
                "doc_values": True,
            },
            "tenant_id": {
                "type": "keyword",
                "doc_values": True,
            },
            "project_id": {
                "type": "keyword",
                "doc_values": True,
            },
            "pipeline_version": {
                "type": "keyword",
                "doc_values": True,
            },
            "content_hash": {
                "type": "keyword",
                "doc_values": True,
            },
            # ===================================================================
            # Document-level Metadata (extracted from first pages)
            # ===================================================================
            "device_name": {
                "type": "keyword",
                "doc_values": True,
                # 장비명 (e.g., "SUPRA XP", "EFEM", "RFID")
            },
            "doc_description": {
                "type": "text",
                "index": False,  # Stored but not searched
                # 문서 설명 (1~2 sentences)
            },
            "chapter": {
                "type": "keyword",
                "doc_values": True,
                # 챕터/섹션 제목 (carry-forward from headings)
            },
            "chunk_summary": {
                "type": "text",
                "index": True,  # Searchable for BM25 (separate from search_text)
                # 청크별 요약 (1~2 sentences)
            },
            "chunk_keywords": {
                "type": "keyword",
                "doc_values": True,
                # 청크별 키워드 (필터링/집계용)
                "fields": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                },
            },
            # ===================================================================
            # Optional Fields
            # ===================================================================
            "page_image_path": {
                "type": "keyword",
                "index": False,  # Not searchable, just stored
            },
            "bbox": {
                "type": "object",
                "enabled": False,  # Stored but not indexed
                # Expected format: {"x": 0, "y": 0, "width": 100, "height": 50}
            },
            "quality_score": {
                "type": "float",
            },
            "summary": {
                "type": "text",
                "index": False,  # Stored for retrieval, not separately searched
            },
            "caption": {
                "type": "text",
                "index": False,
            },
            "tags": {
                "type": "keyword",
                "doc_values": True,
            },
            # ===================================================================
            # Timestamps
            # ===================================================================
            "created_at": {
                "type": "date",
            },
            "updated_at": {
                "type": "date",
            },
            # ===================================================================
            # RAPTOR Hierarchical RAG Fields
            # ===================================================================
            "partition_key": {
                "type": "keyword",
                "doc_values": True,
                # Composite key: device_name + doc_type (e.g., "SUPRA_XP_sop")
            },
            "raptor_level": {
                "type": "integer",
                # 0=leaf, 1,2,3=summary levels
            },
            "raptor_parent_id": {
                "type": "keyword",
                "doc_values": True,
                # Parent node ID in RAPTOR tree
            },
            "raptor_children_ids": {
                "type": "keyword",
                "doc_values": True,
                # Array of child node IDs
            },
            "cluster_id": {
                "type": "keyword",
                "doc_values": True,
                # GMM cluster ID within partition
            },
            "is_summary_node": {
                "type": "boolean",
                # True for summary nodes, False for leaf chunks
            },
            "validation_score": {
                "type": "float",
                # NLI-based summary validation score (0-1)
            },
            "evidence_links": {
                "type": "object",
                "enabled": False,
                # Sentence -> source leaf IDs mapping (stored but not indexed)
            },
            "group_edges": {
                "type": "nested",
                # Soft membership edges to multiple groups
                "properties": {
                    "leaf_id": {"type": "keyword"},
                    "group_id": {"type": "keyword"},
                    "weight": {"type": "float"},
                    "edge_type": {"type": "keyword"},
                    "score": {"type": "float"},
                    "created_at": {"type": "date"},
                },
            },
        },
    }


def get_index_settings(
    number_of_shards: int = 1,
    number_of_replicas: int = 0,
) -> dict[str, Any]:
    """Get index settings.

    Args:
        number_of_shards: Number of primary shards (default: 1 for dev)
        number_of_replicas: Number of replica shards (default: 0 for dev)

    Returns:
        Elasticsearch index settings
    """
    return {
        "number_of_shards": number_of_shards,
        "number_of_replicas": number_of_replicas,
        "refresh_interval": "1s",
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["nori_readingform", "lowercase"],
                }
            }
        },
    }


def get_index_meta(
    embedding_model: str,
    embedding_dim: int | None = None,
    chunking_method: str = "fixed_size",
    chunking_size: int = 512,
    chunking_overlap: int = 50,
    preprocess_method: str = "normalize",
    index_purpose: str = "rag_retrieval",
) -> dict[str, Any]:
    """Get index _meta for pipeline tracking.

    This metadata is stored at the index level, not per document.
    Useful for debugging and operational visibility.

    Args:
        embedding_model: Model name/path used for embeddings
        embedding_dim: Embedding dimensions (optional, for visibility)
        chunking_method: Chunking method name
        chunking_size: Chunk size in characters/tokens
        chunking_overlap: Overlap between chunks
        preprocess_method: Preprocessing method name
        index_purpose: Purpose of this index

    Returns:
        Index _meta definition
    """
    meta = {
        "pipeline": {
            "embedding_model": embedding_model,
            "chunking": {
                "method": chunking_method,
                "size": chunking_size,
                "overlap": chunking_overlap,
            },
            "preprocess": {
                "method": preprocess_method,
            },
            "index_purpose": index_purpose,
        },
    }
    if embedding_dim is not None:
        meta["pipeline"]["embedding_dim"] = embedding_dim
    return meta


# Default mapping with 768 dimensions (BGE-base)
RAG_CHUNKS_MAPPING = get_rag_chunks_mapping(dims=768)


def get_chat_turns_mapping() -> dict[str, Any]:
    """Get chat turns index mapping for conversation history.

    Index naming convention:
        - Index: chat_turns_{env}_v{version}  (e.g., chat_turns_dev_v1)
        - Alias: chat_turns_{env}_current     (e.g., chat_turns_dev_current)

    Each document represents a single turn (user question + assistant answer).
    """
    return {
        "properties": {
            # ===================================================================
            # Session / Turn Identity
            # ===================================================================
            "session_id": {
                "type": "keyword",
                "doc_values": True,
            },
            "turn_id": {
                "type": "integer",
                # Sequential turn number within session (1, 2, 3...)
            },
            "ts": {
                "type": "date",
                # Timestamp when the turn was created
            },
            # ===================================================================
            # Conversation Content
            # ===================================================================
            "user_text": {
                "type": "text",
                "analyzer": "nori",
                # User's question/input
            },
            "assistant_text": {
                "type": "text",
                "analyzer": "nori",
                # Assistant's response
            },
            # ===================================================================
            # Feedback (Satisfaction)
            # ===================================================================
            "feedback_rating": {
                "type": "keyword",
                # "up" | "down"
            },
            "feedback_reason": {
                "type": "text",
                "analyzer": "nori",
                # Optional reason for dissatisfaction
            },
            "feedback_ts": {
                "type": "date",
                # When feedback was recorded
            },
            # ===================================================================
            # Document References (for "이전 1번 문서" feature)
            # ===================================================================
            "doc_refs": {
                "type": "nested",
                "properties": {
                    "slot": {
                        "type": "integer",
                        # User-visible document number (1, 2, 3...)
                    },
                    "doc_id": {
                        "type": "keyword",
                    },
                    "title": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword"},
                        },
                    },
                    "snippet": {
                        "type": "text",
                        # Brief content excerpt shown to user
                    },
                    "page": {
                        "type": "integer",
                    },
                    "pages": {
                        "type": "integer",
                    },
                    "score": {
                        "type": "float",
                    },
                },
            },
            # ===================================================================
            # Session Metadata
            # ===================================================================
            "title": {
                "type": "text",
                "fields": {
                    "raw": {"type": "keyword"},
                },
                # Session title (first message or user-defined)
            },
            "summary": {
                "type": "text",
                # Turn summary for history selection (generated by lightweight LLM)
            },
            "summary_model": {
                "type": "keyword",
                # Model used for summary generation
            },
            "summary_ts": {
                "type": "date",
                # When summary was generated
            },
            # ===================================================================
            # Visibility / Soft Delete
            # ===================================================================
            "is_hidden": {
                "type": "boolean",
                # True = hidden from UI (soft delete), False = visible
                # Default is False (visible)
            },
            # ===================================================================
            # Metadata
            # ===================================================================
            "schema_version": {
                "type": "keyword",
                # For future schema migrations
            },
            "created_at": {
                "type": "date",
            },
            "updated_at": {
                "type": "date",
            },
        },
    }


# Default chat turns mapping
CHAT_TURNS_MAPPING = get_chat_turns_mapping()


__all__ = [
    "get_rag_chunks_mapping",
    "get_index_settings",
    "get_index_meta",
    "get_chat_turns_mapping",
    "RAG_CHUNKS_MAPPING",
    "CHAT_TURNS_MAPPING",
]
