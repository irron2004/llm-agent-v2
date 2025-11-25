"""Application settings using Pydantic Settings.

Configuration is loaded from:
1. Environment variables (highest priority)
2. .env file
3. Default values (lowest priority)
"""

from pydantic import Field
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


class VLLMSettings(BaseSettings):
    """vLLM inference settings."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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


__all__ = [
    "RAGSettings",
    "VLLMSettings",
    "TEISettings",
    "APISettings",
    "rag_settings",
    "vllm_settings",
    "tei_settings",
    "api_settings",
]
