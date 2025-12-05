"""FastAPI dependency providers for heavy objects."""

from functools import lru_cache

from backend.config.settings import rag_settings, vllm_settings
from backend.llm_infrastructure.embedding import get_embedder
from backend.llm_infrastructure.embedding.base import BaseEmbedder
from backend.llm_infrastructure.llm import get_llm
from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.preprocessing import get_preprocessor
from backend.llm_infrastructure.preprocessing.base import BasePreprocessor
from backend.llm_infrastructure.retrieval.base import BaseRetriever


@lru_cache
def get_default_preprocessor() -> BasePreprocessor:
    """Default preprocessor from registry based on settings."""
    return get_preprocessor(
        rag_settings.preprocess_method,
        version=rag_settings.preprocess_version,
    )


@lru_cache
def get_default_embedder() -> BaseEmbedder:
    """Default embedder from registry based on settings."""
    return get_embedder(
        rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )


@lru_cache
def get_default_retriever() -> BaseRetriever:
    """Placeholder retriever until an indexed corpus is wired."""

    class _UnconfiguredRetriever(BaseRetriever):
        def retrieve(self, query: str, top_k: int = 10, **_: object):
            msg = (
                "Default retriever is not configured yet. "
                "Wire a concrete retriever (dense/bm25/hybrid) once a corpus is available."
            )
            raise RuntimeError(msg)

    return _UnconfiguredRetriever()


@lru_cache
def get_default_llm() -> BaseLLM:
    """Default LLM adapter from registry."""
    return get_llm(
        "vllm",
        version="v1",
        base_url=vllm_settings.base_url,
        model=vllm_settings.model_name,
        temperature=vllm_settings.temperature,
        max_tokens=vllm_settings.max_tokens,
        timeout=vllm_settings.timeout,
    )
