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
        default="standard",
        description="Preprocessing method name"
    )
    preprocess_version: str = Field(
        default="v1",
        description="Preprocessing method version"
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


# Global settings instances
rag_settings = RAGSettings()
vllm_settings = VLLMSettings()
tei_settings = TEISettings()
api_settings = APISettings()
deepdoc_settings = DeepDocSettings()
vlm_parser_settings = VlmParserSettings()


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
]
