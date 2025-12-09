"""Application settings using Pydantic Settings.

Configuration is loaded from:
1. Environment variables (highest priority)
2. .env file
3. Default values (lowest priority)
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    """RAG pipeline settings.

    All settings can be overridden via environment variables with prefix RAG_
    Example: RAG_PREPROCESS_METHOD=pe_domain
    """
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Preprocessing
    preprocess_method: str = Field(
        default="normalize",
        description="Preprocessing method name (e.g., normalize, standard, pe_domain)"
    )
    preprocess_version: str = Field(
        default="v1",
        description="Preprocessing method version"
    )
    preprocess_level: str = Field(
        default="L3",
        description="Preprocessing level (used by normalize preprocessor)"
    )

    # Embedding
    embedding_method: str = Field(
        default="bge_base",
        description="Embedding method name"
    )
    embedding_version: str = Field(
        default="v1",
        description="Embedding method version"
    )
    embedding_device: str = Field(
        default="cpu",
        description="Embedding device (cpu/auto/cuda:N)"
    )
    embedding_use_cache: bool = Field(
        default=False,
        description="Use disk cache for embeddings"
    )
    embedding_cache_dir: str = Field(
        default=".cache/embeddings",
        description="Directory for embedding cache"
    )

    # Retrieval
    retrieval_preset: str = Field(
        default="hybrid_rrf_v1",
        description="Retrieval preset name"
    )
    retrieval_method: str = Field(
        default="hybrid",
        description="Retrieval method (dense/bm25/hybrid)"
    )
    retrieval_version: str = Field(
        default="v1",
        description="Retrieval method version"
    )
    retrieval_top_k: int = Field(
        default=10,
        description="Number of documents to retrieve"
    )

    # Hybrid retrieval
    hybrid_dense_weight: float = Field(
        default=0.7,
        description="Dense retrieval weight for hybrid"
    )
    hybrid_sparse_weight: float = Field(
        default=0.3,
        description="Sparse retrieval weight for hybrid"
    )
    hybrid_rrf_k: int = Field(
        default=60,
        description="RRF k parameter"
    )

    # Chunking
    chunking_enabled: bool = Field(
        default=True,
        description="Enable text chunking before embedding"
    )
    chunking_method: str = Field(
        default="fixed_size",
        description="Chunking method (fixed_size, recursive, semantic)"
    )
    chunking_version: str = Field(
        default="v1",
        description="Chunking method version"
    )
    chunk_size: int = Field(
        default=512,
        description="Maximum chunk size (characters or tokens)"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks"
    )
    chunk_split_by: str = Field(
        default="char",
        description="Split unit: 'char' (characters) or 'token' (tokens)"
    )
    chunk_min_size: int = Field(
        default=50,
        description="Minimum chunk size (smaller chunks are merged)"
    )

    # Multi-Query Expansion
    multi_query_enabled: bool = Field(
        default=False,
        description="Enable multi-query expansion for retrieval"
    )
    multi_query_method: str = Field(
        default="llm",
        description="Multi-query expansion method"
    )
    multi_query_n: int = Field(
        default=3,
        description="Number of expanded queries to generate"
    )
    multi_query_include_original: bool = Field(
        default=True,
        description="Include original query in expanded queries"
    )
    multi_query_prompt: str = Field(
        default="general_mq_v1",
        description="Prompt template name for multi-query expansion"
    )

    # Reranking
    rerank_enabled: bool = Field(
        default=False,
        description="Enable reranking of retrieval results"
    )
    rerank_method: str = Field(
        default="cross_encoder",
        description="Reranking method (cross_encoder, llm)"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranking model name/path"
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of results to keep after reranking"
    )

    # Vector store
    vector_store_dir: str = Field(
        default="data/vector_stores",
        description="Directory for persisted vector stores"
    )
    vector_normalize: bool = Field(
        default=True,
        description="L2 normalize vectors for cosine similarity"
    )

    # RAGFlow integration
    ragflow_enabled: bool = Field(
        default=True,
        description="Use RAGFlow for retrieval"
    )
    ragflow_base_url: str = Field(
        default="http://ragflow:9380",
        description="RAGFlow server URL"
    )
    ragflow_api_key: str = Field(
        default="",
        description="RAGFlow API key"
    )
    ragflow_agent_id: str = Field(
        default="",
        description="Default RAGFlow agent ID"
    )


class DeepDocSettings(BaseSettings):
    """DeepDoc vision/OCR/TSR settings."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPDOC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    preferred_backend: str = Field(
        default="RAGFlowPdfParser",
        description="DeepDoc backend: RAGFlowPdfParser, PdfParser, PlainParser",
    )
    model_root: str = Field(
        default="data/deepdoc_models",
        description="Local cache directory for DeepDoc vision models",
    )
    hf_endpoint: str = Field(
        default="",
        description="Custom HuggingFace endpoint (e.g., mirror) for model downloads",
    )
    ocr_model: str = Field(
        default="infiniflow/deepdoc-ocr",
        description="OCR model repo id on HuggingFace",
    )
    layout_model: str = Field(
        default="infiniflow/deepdoc-layout",
        description="Layout recognition model repo id on HuggingFace",
    )
    tsr_model: str = Field(
        default="infiniflow/deepdoc-tsr",
        description="Table structure recognition model repo id on HuggingFace",
    )
    allow_download: bool = Field(
        default=True,
        description="Permit auto-download of models when missing",
    )
    device: str = Field(
        default="cpu",
        description="Preferred device for DeepDoc (cpu/cuda)",
    )


class VlmParserSettings(BaseSettings):
    """Vision-language PDF parser settings (VLM vendor agnostic)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_root: str = Field(
        default="data/deepseek_models",
        description="Local cache directory for VLM models (optional)",
        validation_alias=AliasChoices("VLM_PARSER_MODEL_ROOT", "DEEPSEEK_MODEL_ROOT"),
    )
    model_id: str = Field(
        default="deepseek-ai/deepseek-vl2",
        description="HuggingFace repo id for the VLM",
        validation_alias=AliasChoices("VLM_PARSER_MODEL_ID", "DEEPSEEK_MODEL_ID"),
    )
    prompt: str = Field(
        default="Extract all text on this page as Markdown. Preserve tables and formulas. Do not summarize.",
        description="Default prompt for VLM parsing",
        validation_alias=AliasChoices("VLM_PARSER_PROMPT", "DEEPSEEK_PROMPT"),
    )
    max_new_tokens: int = Field(
        default=2048,
        description="Max new tokens when calling the VLM",
        validation_alias=AliasChoices("VLM_PARSER_MAX_NEW_TOKENS", "DEEPSEEK_MAX_NEW_TOKENS"),
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for VLM generation",
        validation_alias=AliasChoices("VLM_PARSER_TEMPERATURE", "DEEPSEEK_TEMPERATURE"),
    )
    hf_endpoint: str = Field(
        default="",
        description="Custom HuggingFace endpoint for model downloads",
        validation_alias=AliasChoices("VLM_PARSER_HF_ENDPOINT", "DEEPSEEK_HF_ENDPOINT"),
    )
    allow_download: bool = Field(
        default=True,
        description="Permit auto-download of models when missing",
        validation_alias=AliasChoices("VLM_PARSER_ALLOW_DOWNLOAD", "DEEPSEEK_ALLOW_DOWNLOAD"),
    )
    device: str = Field(
        default="cpu",
        description="Preferred device for VLM parsing (cpu/cuda)",
        validation_alias=AliasChoices("VLM_PARSER_DEVICE", "DEEPSEEK_DEVICE"),
    )


class VLLMSettings(BaseSettings):
    """vLLM inference settings."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    base_url: str = Field(
        default="http://vllm:8000",
        description="vLLM server URL"
    )
    model_name: str = Field(
        default="gpt-oss-20b",
        description="Model name/identifier"
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens to generate"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )


class TEISettings(BaseSettings):
    """Text Embeddings Inference settings."""

    model_config = SettingsConfigDict(
        env_prefix="TEI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    endpoint_url: str = Field(
        default="http://tei:80",
        description="TEI server URL"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )


class APISettings(BaseSettings):
    """FastAPI application settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    title: str = Field(
        default="PE Agent API",
        description="API title"
    )
    version: str = Field(
        default="0.1.0",
        description="API version"
    )
    description: str = Field(
        default="PE Agent RAG API",
        description="API description"
    )
    simple_chat_prompt_file: str | None = Field(
        default=None,
        description="Path to system prompt file for simple chat (LLM-only)",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind"
    )
    port: int = Field(
        default=8100,
        description="Port to bind"
    )
    reload: bool = Field(
        default=False,
        description="Auto-reload on code changes"
    )
    log_level: str = Field(
        default="info",
        description="Logging level"
    )


class SearchSettings(BaseSettings):
    """Search service wiring settings."""

    model_config = SettingsConfigDict(
        env_prefix="SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    backend: str = Field(
        default="local",
        description="Search backend type: local | es",
    )
    local_index_path: str = Field(
        default="",
        description="Path to persisted local index (DocumentIndexService.persist_dir)",
    )
    es_host: str = Field(
        default="",
        description="Elasticsearch host (e.g., http://localhost:9200)",
    )
    es_index: str = Field(
        default="",
        description="Elasticsearch index name for vector search",
    )
    es_user: str = Field(
        default="",
        description="Elasticsearch user (optional)",
    )
    es_password: str = Field(
        default="",
        description="Elasticsearch password (optional)",
    )


# Global settings instances
rag_settings = RAGSettings()
vllm_settings = VLLMSettings()
tei_settings = TEISettings()
api_settings = APISettings()
deepdoc_settings = DeepDocSettings()
vlm_parser_settings = VlmParserSettings()
search_settings = SearchSettings()


__all__ = [
    "RAGSettings",
    "VLLMSettings",
    "TEISettings",
    "APISettings",
    "rag_settings",
    "vllm_settings",
    "tei_settings",
    "api_settings",
    "DeepDocSettings",
    "deepdoc_settings",
    "VlmParserSettings",
    "vlm_parser_settings",
    "SearchSettings",
    "search_settings",
]
