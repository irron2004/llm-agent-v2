"""Application settings using Pydantic Settings.

Configuration is loaded from:
1. Environment variables (highest priority)
2. .env file
3. Default values (lowest priority)
"""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .settings_vlm import vlm_settings  # noqa: F401  (optional VLM settings import)


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
        description="Preprocessing method name (e.g., normalize, standard, pe_domain)",
    )
    preprocess_version: str = Field(default="v1", description="Preprocessing method version")
    preprocess_level: str = Field(
        default="L3", description="Preprocessing level (used by normalize preprocessor)"
    )

    # Embedding
    embedding_method: str = Field(default="bge_base", description="Embedding method name")
    embedding_version: str = Field(default="v1", description="Embedding method version")
    embedding_device: str = Field(default="cpu", description="Embedding device (cpu/auto/cuda:N)")
    embedding_use_cache: bool = Field(default=False, description="Use disk cache for embeddings")
    embedding_cache_dir: str = Field(
        default=".cache/embeddings", description="Directory for embedding cache"
    )

    # Retrieval
    retrieval_preset: str = Field(default="hybrid_rrf_v1", description="Retrieval preset name")
    retrieval_method: str = Field(
        default="hybrid", description="Retrieval method (dense/bm25/hybrid)"
    )
    retrieval_version: str = Field(default="v1", description="Retrieval method version")
    prompt_spec_version: str = Field(
        default="v1",
        description=(
            "Prompt specification version for LangGraph YAML templates (RAG_PROMPT_SPEC_VERSION)"
        ),
    )

    retrieval_top_k: int = Field(default=10, description="Number of documents to retrieve")

    llm_method: str = Field(
        default="ollama",
        description="Default LLM method name (e.g., vllm, ollama)",
    )
    llm_version: str = Field(
        default="v1",
        description="Default LLM adapter version",
    )

    # Hybrid retrieval
    hybrid_dense_weight: float = Field(default=0.7, description="Dense retrieval weight for hybrid")
    hybrid_sparse_weight: float = Field(
        default=0.3, description="Sparse retrieval weight for hybrid"
    )
    hybrid_rrf_k: int = Field(default=60, description="RRF k parameter")

    # Chunking
    chunking_enabled: bool = Field(
        default=True, description="Enable text chunking before embedding"
    )
    chunking_method: str = Field(
        default="fixed_size", description="Chunking method (fixed_size, recursive, semantic)"
    )
    chunking_version: str = Field(default="v1", description="Chunking method version")
    chunk_size: int = Field(default=512, description="Maximum chunk size (characters or tokens)")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    chunk_split_by: str = Field(
        default="char", description="Split unit: 'char' (characters) or 'token' (tokens)"
    )
    chunk_min_size: int = Field(
        default=50, description="Minimum chunk size (smaller chunks are merged)"
    )

    # Multi-Query Expansion
    multi_query_enabled: bool = Field(
        default=False, description="Enable multi-query expansion for retrieval"
    )
    multi_query_method: str = Field(default="llm", description="Multi-query expansion method")
    multi_query_n: int = Field(default=3, description="Number of expanded queries to generate")
    multi_query_include_original: bool = Field(
        default=True, description="Include original query in expanded queries"
    )
    multi_query_prompt: str = Field(
        default="general_mq_v1", description="Prompt template name for multi-query expansion"
    )

    # Section expansion (chapter-aware retrieval)
    section_expand_enabled: bool = Field(default=True, description="Enable section expansion")
    section_expand_top_groups: int = Field(default=2, description="Max groups to expand")
    section_expand_max_pages: int = Field(default=20, description="Max pages per group")
    section_expand_allowed_sources: str = Field(
        default="title,toc_match,carry",
        description="Allowed chapter_source values for expansion triggers",
    )

    # RAPTOR
    raptor_enabled: bool = Field(
        default=False, description="Enable RAPTOR hierarchical retrieval"
    )
    raptor_tree_strategy: str = Field(
        default="collapsed",
        description="RAPTOR tree search strategy (collapsed/tree_traversal)",
    )
    raptor_expand_summaries: bool = Field(
        default=True, description="Expand summary nodes to leaf evidence"
    )
    raptor_max_levels: int = Field(
        default=3, description="Maximum RAPTOR tree depth"
    )
    raptor_min_partition_size: int = Field(
        default=10, description="Minimum chunks per partition to build tree"
    )

    # Reranking
    rerank_enabled: bool = Field(default=False, description="Enable reranking of retrieval results")
    rerank_method: str = Field(
        default="cross_encoder", description="Reranking method (cross_encoder, llm)"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Reranking model name/path"
    )
    rerank_top_k: int = Field(default=5, description="Number of results to keep after reranking")

    # Vector store
    vector_store_dir: str = Field(
        default="data/vector_stores", description="Directory for persisted vector stores"
    )
    vector_normalize: bool = Field(
        default=True, description="L2 normalize vectors for cosine similarity"
    )

    # RAGFlow integration
    ragflow_enabled: bool = Field(default=True, description="Use RAGFlow for retrieval")
    ragflow_base_url: str = Field(default="http://ragflow:9380", description="RAGFlow server URL")
    ragflow_api_key: str = Field(default="", description="RAGFlow API key")
    ragflow_agent_id: str = Field(default="", description="Default RAGFlow agent ID")


class DeepDocSettings(BaseSettings):
    """DeepDoc vision/OCR/TSR settings."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPDOC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
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
        default=(
            "Read this page and extract all content as Markdown.\n\n"
            "Rules:\n"
            "- Output ONLY valid Markdown (no LaTeX, no $...$ or $$...$$)\n"
            "- Convert mathematical formulas to plain text or Unicode symbols\n"
            "- Preserve tables using Markdown table syntax (| column |)\n"
            "- Keep headings with # syntax\n"
            "- Preserve lists and bullet points\n"
            "- Do not summarize or omit any text\n"
            "- Do not add explanations or comments\n"
            "- NEVER output repeated characters like ||||, &&&&, ----, ====, etc."
        ),
        description="Default prompt for VLM parsing",
        validation_alias=AliasChoices("VLM_PARSER_PROMPT", "DEEPSEEK_PROMPT"),
    )
    max_new_tokens: int = Field(
        default=9096,
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
        extra="ignore",
    )

    base_url: str = Field(default="http://vllm:8000", description="vLLM server URL")
    model_name: str = Field(default="gpt-oss-20b", description="Model name/identifier")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    reasoning_effort: str | None = Field(
        default=None, description="Reasoning effort hint for reasoning models (low|medium|high)."
    )
    max_tokens: int = Field(default=30000, description="Maximum tokens to generate")
    timeout: int = Field(default=60, description="Request timeout in seconds")


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL (native API base or OpenAI-compatible /v1 base)",
    )
    model_name: str = Field(default="qwen2.5:14b", description="Model name/identifier")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=30000, description="Maximum tokens to generate")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    repeat_penalty: float = Field(
        default=1.3, description="Repetition penalty (1.0=off, >1.0=penalize)"
    )
    repeat_last_n: int = Field(
        default=256,
        description="Token window for repeat penalty (default ollama=64, 0=disabled, -1=ctx_size)",
    )
    num_ctx: int = Field(
        default=131072,
        description="Context window size in tokens (default ollama=8192, 131072=128K)",
    )


class VlmClientSettings(BaseSettings):
    """VLM (Vision-Language Model) client settings for OpenAI-compatible API.

    Used for connecting to vLLM-served Qwen-VL or similar vision models.
    """

    model_config = SettingsConfigDict(
        env_prefix="VLM_CLIENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="VLM server base URL (OpenAI-compatible API)",
    )
    model: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        description="VLM model name/identifier",
    )
    timeout: int = Field(
        default=1200,
        description="Request timeout in seconds (VLM inference can be slow)",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens to generate per page",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0.0 for deterministic)",
    )


class IngestSettings(BaseSettings):
    """Document ingestion pipeline settings.

    Controls metadata extraction, chapter assignment, and summarization.
    """

    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Metadata extraction
    enable_doc_metadata: bool = Field(
        default=True,
        description="Extract device_name/doc_description from first pages",
    )
    enable_chapter_extraction: bool = Field(
        default=True,
        description="Assign chapter titles with carry-forward logic",
    )
    enable_summarization: bool = Field(
        default=False,
        description="Generate chunk-level summaries (requires text LLM)",
    )
    use_llm_fallback: bool = Field(
        default=True,
        description="Use LLM when rule-based extraction fails",
    )

    # Chunk settings
    max_chunk_size: int = Field(
        default=2000,
        description="Maximum chunk size before additional splitting (chars)",
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size for summarization",
    )

    # First pages for document metadata
    doc_metadata_pages: int = Field(
        default=3,
        description="Number of first pages to analyze for doc metadata",
    )


class TEISettings(BaseSettings):
    """Text Embeddings Inference settings."""

    model_config = SettingsConfigDict(
        env_prefix="TEI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    endpoint_url: str = Field(default="http://tei:80", description="TEI server URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class APISettings(BaseSettings):
    """FastAPI application settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    title: str = Field(default="PE Agent API", description="API title")
    version: str = Field(default="0.1.0", description="API version")
    description: str = Field(default="PE Agent RAG API", description="API description")
    simple_chat_prompt_file: str | None = Field(
        default=None,
        description="Path to system prompt file for simple chat (LLM-only)",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8100, description="Port to bind")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    log_level: str = Field(default="info", description="Logging level")
    log_to_file: bool = Field(default=False, description="Write API logs to file")
    log_file_path: str = Field(default="/data/logs/api/api.log", description="API log file path")
    log_max_bytes: int = Field(
        default=20 * 1024 * 1024, description="Max bytes per log file before rotation"
    )
    log_backup_count: int = Field(default=5, description="Number of rotated log files to keep")
    enable_legacy_compat_routes: bool = Field(
        default=False,
        description="Enable legacy compatibility endpoints (e.g. /api/search/chat-pipeline)",
    )


class SummarizationSettings(BaseSettings):
    """Document summarization settings."""

    model_config = SettingsConfigDict(
        env_prefix="SUMMARIZATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_method: str = Field(
        default="vllm",
        description="LLM method for summarization",
    )
    llm_version: str = Field(
        default="v1",
        description="LLM method version",
    )
    chunk_size: int = Field(
        default=900,
        ge=100,
        le=4000,
        description="Default chunk size for document splitting",
    )
    chunk_overlap: int = Field(
        default=120,
        ge=0,
        le=500,
        description="Overlap between chunks",
    )
    prompt_version: str = Field(
        default="v1",
        description="Prompt template version",
    )


class MinioSettings(BaseSettings):
    """MinIO object storage settings for document images."""

    model_config = SettingsConfigDict(
        env_prefix="MINIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    endpoint: str = Field(
        default="http://minio:9000",
        description="MinIO server endpoint URL",
    )

    @model_validator(mode="after")
    def _resolve_minio_host(self) -> "MinioSettings":
        """Replace Docker-internal 'minio' hostname with 'localhost' when running on host."""
        if "://minio:" in self.endpoint and not Path("/.dockerenv").exists():
            import logging

            logger = logging.getLogger(__name__)
            original = self.endpoint
            resolved = self.endpoint.replace("://minio:", "://localhost:")
            object.__setattr__(self, "endpoint", resolved)
            logger.warning(
                "MinIO endpoint auto-resolved: %s -> %s (not running in Docker)",
                original,
                resolved,
            )
        return self
    access_key: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("MINIO_ACCESS_KEY", "MINIO_ROOT_USER"),
        description="MinIO access key (root user)",
    )
    secret_key: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("MINIO_SECRET_KEY", "MINIO_ROOT_PASSWORD"),
        description="MinIO secret key (root password)",
    )
    bucket: str = Field(
        default="doc-images",
        description="Default bucket name for document images",
    )
    secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO connection",
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
        default="http://localhost:9200",
        description="Elasticsearch host (e.g., http://localhost:9200)",
    )
    es_index: str = Field(
        default="",
        description=(
            "Elasticsearch index name for vector search (legacy, use es_index_* for new setup)"
        ),
    )
    es_user: str = Field(
        default="",
        description="Elasticsearch user (optional)",
    )
    es_password: str = Field(
        default="",
        description="Elasticsearch password (optional)",
    )
    # ES Index Management Settings (SU-2509)
    es_env: str = Field(
        default="dev",
        description="Environment name for index naming (dev, staging, prod)",
    )
    es_index_prefix: str = Field(
        default="rag_chunks",
        description="Index name prefix (e.g., rag_chunks -> rag_chunks_dev_v1)",
    )
    es_index_version: int = Field(
        default=1,
        description="Current index version number",
    )
    es_embedding_dims: int = Field(
        default=768,
        description="Embedding vector dimensions (768 for BGE-base, 1024 for KoE5/multilingual-e5)",
    )
    chunk_version: Literal["v2", "v3"] = Field(
        default="v2",
        description=(
            "Search runtime chunk layout version (v2=single index alias, v3=content+embed split)"
        ),
    )
    v2_alias: str = Field(
        default="",
        description=(
            "Optional alias override for v2 search index "
            "(default: {es_index_prefix}_{es_env}_current)"
        ),
    )
    v3_content_index: str = Field(
        default="chunk_v3_content",
        description="Content index name for v3 split-index retrieval",
    )
    v3_embed_index: str = Field(
        default="",
        description="Embedding index name for v3 split-index retrieval",
    )
    v3_embed_model_key: str = Field(
        default="",
        description="Optional v3 embed model key used to build index name chunk_v3_embed_{key}_v1",
    )


class AgentSettings(BaseSettings):
    """Agent settings."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mq_mode_default: Literal["off", "fallback", "on"] = Field(
        default="fallback",
        description="Default mq_mode when request does not provide one",
    )
    second_stage_doc_retrieve_enabled: bool = False
    early_page_penalty_enabled: bool = True
    early_page_penalty_max_page: int = 2
    early_page_penalty_factor: float = 0.3
    second_stage_max_doc_ids: int = 1
    second_stage_top_k: int = 50
    sop_soft_boost_factor: float = Field(
        default=1.30,
        gt=0.0,
        le=5.0,
        description="Score multiplier for SOP docs when SOP intent detected (1.0 = no boost)",
    )
    procedure_boost_enabled: bool = True
    procedure_boost_factor: float = Field(
        default=1.40,
        gt=0.0,
        le=5.0,
        description="Score multiplier for Work Procedure/Flow Chart pages when query contains procedure keywords",
    )
    scope_penalty_enabled: bool = True
    scope_penalty_factor: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Score multiplier for Scope/Contents/TOC pages (penalize non-procedure pages)",
    )
    device_aliases: dict[str, list[str]] = Field(
        default={
            "SUPRA XP": ["SUPRA XP", "ZEDIUS XP"],
            "ZEDIUS XP": ["SUPRA XP", "ZEDIUS XP"],
            "SUPRA V+": ["SUPRA V+", "SUPRA Vplus"],
            "SUPRA VP": ["SUPRA VP", "SUPRA Vplus"],
            "SUPRA Vp": ["SUPRA Vp", "SUPRA Vplus"],
            "SUPRA Vplus": ["SUPRA Vplus", "SUPRA V+", "SUPRA VP", "SUPRA Vp"],
        },
        description="Device name aliases: key → list of equivalent device names for search expansion",
    )
    abbreviation_expand_enabled: bool = Field(
        default=True,
        description="Enable domain dictionary abbreviation expansion in queries",
    )
    abbreviation_dict_path: str = Field(
        default="data/semicon_word.json",
        description="Path to semicon_word.json domain dictionary for abbreviation expansion",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum RRF score threshold (0.0-1.0). Documents below this score are filtered out. 0.0 = no filtering.",
    )
    dedupe_by_doc_id: bool = Field(
        default=True,
        description="Deduplicate results by base doc_id, keeping only the highest-scoring page per document.",
    )
    doc_type_diversity_enabled: bool = Field(
        default=True,
        description="SOP+Setup 동시 선택 시 top-k 결과에 doc_type 다양성 쿼터 적용.",
    )
    doc_type_diversity_min_setup: int = Field(
        default=3,
        description="SOP+Setup 동시 선택 시 top-k에 포함할 최소 setup 문서 수 (가용 시).",
    )
    doc_type_diversity_min_sop: int = Field(
        default=3,
        description="SOP+Setup 동시 선택 시 top-k에 포함할 최소 SOP 문서 수 (가용 시).",
    )


# Global settings instances
rag_settings = RAGSettings()
vllm_settings = VLLMSettings()
ollama_settings = OllamaSettings()
vlm_client_settings = VlmClientSettings()
tei_settings = TEISettings()
api_settings = APISettings()
deepdoc_settings = DeepDocSettings()
vlm_parser_settings = VlmParserSettings()
search_settings = SearchSettings()
ingest_settings = IngestSettings()
summarization_settings = SummarizationSettings()
minio_settings = MinioSettings()
agent_settings = AgentSettings()


__all__ = [
    "RAGSettings",
    "VLLMSettings",
    "OllamaSettings",
    "VlmClientSettings",
    "TEISettings",
    "APISettings",
    "IngestSettings",
    "SummarizationSettings",
    "rag_settings",
    "vllm_settings",
    "ollama_settings",
    "vlm_client_settings",
    "tei_settings",
    "api_settings",
    "ingest_settings",
    "summarization_settings",
    "DeepDocSettings",
    "deepdoc_settings",
    "VlmParserSettings",
    "vlm_parser_settings",
    "SearchSettings",
    "search_settings",
    "MinioSettings",
    "minio_settings",
    "AgentSettings",
    "agent_settings",
]
