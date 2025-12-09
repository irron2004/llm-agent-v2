"""FastAPI dependency providers for heavy objects."""

from functools import lru_cache
from pathlib import Path
from typing import Callable

from backend.config.settings import api_settings, rag_settings, vllm_settings
from backend.llm_infrastructure.embedding import get_embedder
from backend.llm_infrastructure.embedding.base import BaseEmbedder
from backend.llm_infrastructure.llm import get_llm
from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.preprocessing import get_preprocessor
from backend.llm_infrastructure.preprocessing.base import BasePreprocessor
from backend.llm_infrastructure.reranking import get_reranker as _get_reranker
from backend.llm_infrastructure.reranking.base import BaseReranker
from backend.llm_infrastructure.retrieval.base import BaseRetriever
from backend.services.chat_service import ChatService
from backend.services.rag_service import RAGService, RAGResponse
from backend.services.search_service import SearchService


@lru_cache
def get_default_preprocessor() -> BasePreprocessor:
    """Default preprocessor from registry based on settings."""
    return get_preprocessor(
        rag_settings.preprocess_method,
        version=rag_settings.preprocess_version,
        level=rag_settings.preprocess_level,
    )


def get_preprocessor_factory() -> Callable[[str | None], BasePreprocessor]:
    """Provide a factory to fetch preprocessors per request (level override support)."""
    default = get_default_preprocessor()

    def factory(level: str | None = None) -> BasePreprocessor:
        """
        요청 본문에 들어온 level 값을 보고 전처리기를 고르려면, 본문을 읽은 뒤
        에야 선택할 수 있어서 DI 한 번으로 끝내기 어렵습니다. 그래서:

        - 기본값은 캐시된 전처리기(설정 레벨)를 재사용하고,
        - 본문에서 다른 level이 오면 그때만 새 인스턴스를 만들어 주도록,
        - 라우터 안에서 body.level을 보고 고를 수 있게 팩토리(클로저)로 감쌌습
            니다.

        그냥 get_preprocessor()를 바로 DI로 받으면 요청별 level을 반영할 수 없
        고, 매 요청마다 새로 만들거나 이중 DI가 필요해집니다. 만약 단순히 라우
        터에서 get_preprocessor(method, version, level=body.level)를 직접 호출
        하는 쪽이 더 낫다면 그렇게 바꿔도 됩니다만, 기본 캐시 재사용 + override
        둘 다 잡으려고 factory 패턴을 쓴 것입니다.
        """
        if level is None or level == rag_settings.preprocess_level:
            return default
        return get_preprocessor(
            rag_settings.preprocess_method,
            version=rag_settings.preprocess_version,
            level=level,
        )

    return factory


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


class _NotConfiguredSearchService:
    """Fallback search service that raises a helpful error."""

    def search(self, *_: object, **__: object):
        msg = (
            "Search service is not configured. "
            "Provide a concrete SearchService via dependency override or wire an indexed corpus."
        )
        raise RuntimeError(msg)


class _NotConfiguredRAGService:
    """Fallback RAG service that raises a helpful error."""

    def query(self, *_: object, **__: object) -> RAGResponse:  # pragma: no cover - simple guard
        msg = (
            "RAG service is not configured. "
            "Provide a concrete RAGService via dependency override or wire an indexed corpus."
        )
        raise RuntimeError(msg)


@lru_cache
def get_search_service() -> SearchService:
    """Provide a search service (override in tests or when a corpus is available)."""
    return _NotConfiguredSearchService()  # type: ignore[return-value]


@lru_cache
def get_rag_service() -> RAGService:
    """Provide a RAG service (override in tests or when a corpus is available)."""
    return _NotConfiguredRAGService()  # type: ignore[return-value]


@lru_cache
def get_chat_service() -> ChatService:
    """Provide a chat service (LLM only, no retrieval)."""
    return ChatService()


@lru_cache
def get_simple_chat_prompt() -> str | None:
    """Load system prompt for simple chat from file if configured."""
    path = api_settings.simple_chat_prompt_file
    if not path:
        return None
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Simple chat prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


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


@lru_cache
def get_reranker() -> BaseReranker:
    """Provide a reranker based on settings."""
    return _get_reranker(
        rag_settings.rerank_method,
        version="v1",
        model_name=rag_settings.rerank_model,
        device=rag_settings.embedding_device,
    )
