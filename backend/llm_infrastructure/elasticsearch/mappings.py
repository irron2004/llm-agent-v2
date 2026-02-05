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
            "edited": {
                "type": "boolean",
                # True if this turn was created from an edited user message
            },
            "parent_session_id": {
                "type": "keyword",
                # Parent session ID if this session is branched
            },
            "branched_from_turn_id": {
                "type": "integer",
                # Turn ID where the branch was created
            },
            "is_branch": {
                "type": "boolean",
                # True if this session is a branch
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


def get_feedback_mapping() -> dict[str, Any]:
    """Get feedback index mapping for detailed feedback storage.

    Index naming convention:
        - Index: feedback_{env}_v{version}  (e.g., feedback_dev_v1)
        - Alias: feedback_{env}_current     (e.g., feedback_dev_current)

    Each document represents feedback for a single turn with detailed scores.
    Designed for LLM fine-tuning data collection.
    """
    return {
        "properties": {
            # ===================================================================
            # Reference Keys (FK to chat_turns)
            # ===================================================================
            "session_id": {
                "type": "keyword",
                "doc_values": True,
            },
            "turn_id": {
                "type": "integer",
            },
            # ===================================================================
            # Conversation Content (copied for join-free retrieval)
            # ===================================================================
            "user_text": {
                "type": "text",
                "analyzer": "nori",
            },
            "assistant_text": {
                "type": "text",
                "analyzer": "nori",
            },
            # ===================================================================
            # Detailed Feedback Scores (1-5)
            # ===================================================================
            "accuracy": {
                "type": "integer",
                # 답변이 사실적으로 정확한가 (1-5)
            },
            "completeness": {
                "type": "integer",
                # 답변이 질문에 충분히 답했는가 (1-5)
            },
            "relevance": {
                "type": "integer",
                # 답변이 질문과 관련이 있는가 (1-5)
            },
            "avg_score": {
                "type": "float",
                # 평균 점수 (집계용)
            },
            "rating": {
                "type": "keyword",
                # "up" | "down" (하위 호환, avg >= 3 이면 up)
            },
            # ===================================================================
            # Free-form Feedback
            # ===================================================================
            "comment": {
                "type": "text",
                "analyzer": "nori",
                # 자유 의견 (선택사항)
            },
            # ===================================================================
            # Reviewer Info
            # ===================================================================
            "reviewer_name": {
                "type": "keyword",
                # 피드백 제출자 이름 (선택, 집계용)
            },
            # ===================================================================
            # Execution Logs (at feedback time)
            # ===================================================================
            "logs": {
                "type": "text",
                "index": False,
                # 피드백 시점의 실행 로그 (저장만, 검색 불필요)
            },
            # ===================================================================
            # Timestamps
            # ===================================================================
            "ts": {
                "type": "date",
                # 피드백 제출 시간
            },
            "created_at": {
                "type": "date",
            },
            "updated_at": {
                "type": "date",
            },
        },
    }


# Default feedback mapping
FEEDBACK_MAPPING = get_feedback_mapping()


def get_retrieval_evaluation_mapping() -> dict[str, Any]:
    """Get retrieval evaluation index mapping for query-unit relevance scoring.

    Index naming convention:
        - Index: retrieval_evaluations_{env}_v{version}  (e.g., retrieval_evaluations_dev_v1)
        - Alias: retrieval_evaluations_{env}_current     (e.g., retrieval_evaluations_dev_current)

    Each document represents a query-unit evaluation containing multiple document relevance scores.
    Used for retrieval test set creation and search parameter tuning.

    Storage structure:
        {
            "query_id": "sess1:turn1",  # PK (chat: session:turn, search: search:timestamp)
            "query": "원본 쿼리",
            "relevant_docs": ["doc_001", "doc_003"],      # 자동 생성 (score >= 3)
            "irrelevant_docs": ["doc_002", "doc_004"],    # 자동 생성 (score < 3)
            "doc_details": [{ ... }]                      # 필수, 개별 문서 점수
        }
    """
    return {
        "properties": {
            # ===================================================================
            # Primary Key
            # ===================================================================
            "query_id": {
                "type": "keyword",
                "doc_values": True,
                # PK: chat="{session_id}:{turn_id}", search="search:{timestamp}"
            },
            # ===================================================================
            # Source Context
            # ===================================================================
            "source": {
                "type": "keyword",
                "doc_values": True,
                # "chat" or "search"
            },
            "session_id": {
                "type": "keyword",
                "doc_values": True,
                # Chat only: session reference
            },
            "turn_id": {
                "type": "integer",
                # Chat only: turn reference
            },
            # ===================================================================
            # Query Information
            # ===================================================================
            "query": {
                "type": "text",
                "analyzer": "nori",
                "fields": {
                    "raw": {"type": "keyword"},
                },
                # Original user query
            },
            # ===================================================================
            # Aggregated Document Lists (auto-generated from doc_details)
            # ===================================================================
            "relevant_docs": {
                "type": "keyword",
                "doc_values": True,
                # Doc IDs with relevance_score >= 3
            },
            "irrelevant_docs": {
                "type": "keyword",
                "doc_values": True,
                # Doc IDs with relevance_score < 3
            },
            # ===================================================================
            # Document Details (nested, required)
            # ===================================================================
            "doc_details": {
                "type": "nested",
                "properties": {
                    "doc_id": {
                        "type": "keyword",
                        "doc_values": True,
                    },
                    "chunk_id": {
                        "type": "keyword",
                        "doc_values": True,
                    },
                    "doc_rank": {
                        "type": "integer",
                        # 1-based rank in search results
                    },
                    "doc_title": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword"},
                        },
                    },
                    "doc_snippet": {
                        "type": "text",
                        "index": False,  # Stored but not searched
                    },
                    "page": {
                        "type": "integer",
                    },
                    "relevance_score": {
                        "type": "integer",
                        # Human-evaluated relevance score (1-5)
                    },
                    "retrieval_score": {
                        "type": "float",
                        # Original search score from retrieval engine
                    },
                },
            },
            # ===================================================================
            # Search Parameters (for reproducibility, Search page only)
            # ===================================================================
            "search_params": {
                "type": "object",
                "enabled": False,  # Stored but not indexed
                # Contains: field_weights, bm25_only, dense_weight, sparse_weight, size, etc.
            },
            # ===================================================================
            # Filter Context (for search reproducibility)
            # ===================================================================
            "filter_devices": {
                "type": "keyword",
                "doc_values": True,
                # Device filter used during search
            },
            "filter_doc_types": {
                "type": "keyword",
                "doc_values": True,
                # Document type filter used during search
            },
            "search_queries": {
                "type": "text",
                "analyzer": "nori",
                # Multi-query expansion results (for debugging retrieval issues)
            },
            # ===================================================================
            # Reviewer Info
            # ===================================================================
            "reviewer_name": {
                "type": "keyword",
                # Evaluator name (optional, for tracking)
            },
            # ===================================================================
            # Timestamps
            # ===================================================================
            "ts": {
                "type": "date",
                # Last evaluation timestamp
            },
            "created_at": {
                "type": "date",
            },
            "updated_at": {
                "type": "date",
            },
        },
    }


# Default retrieval evaluation mapping
RETRIEVAL_EVALUATION_MAPPING = get_retrieval_evaluation_mapping()


def get_batch_answer_runs_mapping() -> dict[str, Any]:
    """Get batch answer runs index mapping for batch answer generation metadata.

    Index naming convention:
        - Index: batch_answer_runs_{env}_v{version}  (e.g., batch_answer_runs_dev_v1)
        - Alias: batch_answer_runs_{env}_current     (e.g., batch_answer_runs_dev_current)

    Each document represents a batch answer generation run configuration and status.
    """
    return {
        "properties": {
            # ===================================================================
            # Run Identity
            # ===================================================================
            "run_id": {
                "type": "keyword",
                "doc_values": True,
                # UUID for this run
            },
            "name": {
                "type": "text",
                "fields": {
                    "raw": {"type": "keyword"},
                },
                # User-defined run name (e.g., "RRF k=60 테스트")
            },
            "description": {
                "type": "text",
                "analyzer": "nori",
                # Optional description
            },
            # ===================================================================
            # Status
            # ===================================================================
            "status": {
                "type": "keyword",
                "doc_values": True,
                # "pending" | "running" | "completed" | "failed" | "cancelled"
            },
            "progress": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer"},
                    "completed": {"type": "integer"},
                    "failed": {"type": "integer"},
                },
            },
            "error_message": {
                "type": "text",
                "index": False,
                # Error message if status is "failed"
            },
            # ===================================================================
            # Source Configuration
            # ===================================================================
            "source_type": {
                "type": "keyword",
                "doc_values": True,
                # "retrieval_test" - 현재는 retrieval test 결과만 지원
            },
            "source_run_id": {
                "type": "keyword",
                "doc_values": True,
                # Reference to retrieval test run (if applicable)
            },
            "source_config": {
                "type": "object",
                "enabled": False,  # Stored but not indexed
                # Search config snapshot: use_rrf, rrf_k, rerank, dense_weight, sparse_weight, etc.
            },
            # ===================================================================
            # LLM Configuration
            # ===================================================================
            "llm_config": {
                "type": "object",
                "enabled": False,
                # LLM settings: model, temperature, max_tokens, etc.
            },
            # ===================================================================
            # Aggregated Metrics (updated on completion)
            # ===================================================================
            "metrics": {
                "type": "object",
                "properties": {
                    "avg_rating": {"type": "float"},
                    "rating_count": {"type": "integer"},
                    "avg_latency_ms": {"type": "float"},
                    "total_tokens": {"type": "integer"},
                },
            },
            # ===================================================================
            # Timestamps
            # ===================================================================
            "started_at": {
                "type": "date",
            },
            "completed_at": {
                "type": "date",
            },
            "created_at": {
                "type": "date",
            },
            "updated_at": {
                "type": "date",
            },
        },
    }


# Default batch answer runs mapping
BATCH_ANSWER_RUNS_MAPPING = get_batch_answer_runs_mapping()


def get_batch_answer_results_mapping() -> dict[str, Any]:
    """Get batch answer results index mapping for individual answer results.

    Index naming convention:
        - Index: batch_answer_results_{env}_v{version}  (e.g., batch_answer_results_dev_v1)
        - Alias: batch_answer_results_{env}_current     (e.g., batch_answer_results_dev_current)

    Each document represents a single question's answer result within a batch run.
    """
    return {
        "properties": {
            # ===================================================================
            # Result Identity
            # ===================================================================
            "result_id": {
                "type": "keyword",
                "doc_values": True,
                # UUID for this result
            },
            "run_id": {
                "type": "keyword",
                "doc_values": True,
                # FK to batch_answer_runs
            },
            # ===================================================================
            # Question Information
            # ===================================================================
            "question_id": {
                "type": "keyword",
                "doc_values": True,
                # Original question ID from source
            },
            "question": {
                "type": "text",
                "analyzer": "nori",
                "fields": {
                    "raw": {"type": "keyword"},
                },
            },
            "category": {
                "type": "keyword",
                "doc_values": True,
                # Question category (optional)
            },
            # ===================================================================
            # Answer Content
            # ===================================================================
            "answer": {
                "type": "text",
                "analyzer": "nori",
                # Generated answer text
            },
            "reasoning": {
                "type": "text",
                "index": False,
                # LLM reasoning/chain-of-thought (stored but not searched)
            },
            # ===================================================================
            # Search Results Used (nested for per-doc details)
            # ===================================================================
            "search_results": {
                "type": "nested",
                "properties": {
                    "rank": {"type": "integer"},
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "snippet": {
                        "type": "text",
                        "index": False,
                    },
                    "content": {
                        "type": "text",
                        "index": False,  # Full content for LLM context
                    },
                    "score": {"type": "float"},
                    "page": {"type": "integer"},
                    "device_name": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                },
            },
            "search_result_count": {
                "type": "integer",
                # Number of search results used
            },
            # ===================================================================
            # Ground Truth (for metrics calculation)
            # ===================================================================
            "ground_truth_doc_ids": {
                "type": "keyword",
                "doc_values": True,
                # Expected relevant doc IDs
            },
            # ===================================================================
            # Metrics
            # ===================================================================
            "retrieval_metrics": {
                "type": "object",
                "properties": {
                    "hit_at_1": {"type": "boolean"},
                    "hit_at_3": {"type": "boolean"},
                    "hit_at_5": {"type": "boolean"},
                    "hit_at_10": {"type": "boolean"},
                    "reciprocal_rank": {"type": "float"},
                    "first_relevant_rank": {"type": "integer"},
                },
            },
            "latency_ms": {
                "type": "integer",
                # Answer generation latency
            },
            "token_count": {
                "type": "object",
                "properties": {
                    "input": {"type": "integer"},
                    "output": {"type": "integer"},
                },
            },
            # ===================================================================
            # Human Evaluation (optional, filled later)
            # ===================================================================
            "rating": {
                "type": "integer",
                # Human rating 1-5
            },
            "rating_comment": {
                "type": "text",
                "index": False,
            },
            "rated_by": {
                "type": "keyword",
            },
            "rated_at": {
                "type": "date",
            },
            # ===================================================================
            # Status
            # ===================================================================
            "status": {
                "type": "keyword",
                "doc_values": True,
                # "pending" | "completed" | "failed"
            },
            "error_message": {
                "type": "text",
                "index": False,
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
        },
    }


# Default batch answer results mapping
BATCH_ANSWER_RESULTS_MAPPING = get_batch_answer_results_mapping()


__all__ = [
    "get_rag_chunks_mapping",
    "get_index_settings",
    "get_index_meta",
    "get_chat_turns_mapping",
    "get_feedback_mapping",
    "get_retrieval_evaluation_mapping",
    "get_batch_answer_runs_mapping",
    "get_batch_answer_results_mapping",
    "RAG_CHUNKS_MAPPING",
    "CHAT_TURNS_MAPPING",
    "FEEDBACK_MAPPING",
    "RETRIEVAL_EVALUATION_MAPPING",
    "BATCH_ANSWER_RUNS_MAPPING",
    "BATCH_ANSWER_RESULTS_MAPPING",
]
