"""Integrated RAG service combining retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.config.settings import rag_settings
from backend.llm_infrastructure.preprocessing import get_preprocessor
from backend.llm_infrastructure.preprocessing.base import BasePreprocessor
from backend.services.embedding_service import EmbeddingService
from backend.services.document_service import (
    DocumentIndexService,
    IndexedCorpus,
    SourceDocument,
)
from backend.services.search_service import SearchService
from backend.services.chat_service import ChatService
from backend.llm_infrastructure.retrieval.base import RetrievalResult


@dataclass
class RAGResponse:
    """Response from RAG query with answer and context."""

    answer: str
    context: list[RetrievalResult]
    question: str
    metadata: dict | None = None


class RAGService:
    """End-to-end RAG pipeline: preprocessing → retrieval → generation."""

    def __init__(
        self,
        corpus: IndexedCorpus,
        *,
        preprocessor: Optional[BasePreprocessor] = None,
        search_service: Optional[SearchService] = None,
        chat_service: Optional[ChatService] = None,
        retrieval_top_k: Optional[int] = None,
        context_template: Optional[str] = None,
    ) -> None:
        """Initialize RAG service with corpus and optional overrides.

        Args:
            corpus: Indexed document corpus for retrieval
            preprocessor: Text preprocessor (default: from settings)
            search_service: Search service (default: created from corpus)
            chat_service: Chat service (default: created with vllm)
            retrieval_top_k: Number of docs to retrieve (default: from settings)
            context_template: Template for context prompt (default: simple format)
        """
        self.corpus = corpus
        self.retrieval_top_k = retrieval_top_k or rag_settings.retrieval_top_k

        # Preprocessor
        if preprocessor is None:
            self.preprocessor = get_preprocessor(
                rag_settings.preprocess_method,
                version=rag_settings.preprocess_version,
            )
        else:
            self.preprocessor = preprocessor

        # Search service
        if search_service is None:
            self.search_service = SearchService(
                corpus,
                method=rag_settings.retrieval_method,
                version=rag_settings.retrieval_version,
                top_k=self.retrieval_top_k,
            )
        else:
            self.search_service = search_service

        # Chat service
        if chat_service is None:
            self.chat_service = ChatService()
        else:
            self.chat_service = chat_service

        # Context template
        if context_template is None:
            self.context_template = (
                "Answer the question based on the following context:\n\n"
                "{context}\n\n"
                "Question: {question}"
            )
        else:
            self.context_template = context_template

    @classmethod
    def from_settings(cls, corpus: IndexedCorpus) -> "RAGService":
        """Create RAG service using global settings."""
        return cls(corpus)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query text."""
        if self.preprocessor is None:
            return query
        processed = list(self.preprocessor.preprocess([query]))
        return processed[0] if processed else query

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.content}")
        return "\n\n".join(context_parts)

    def _build_system_prompt(self, context: str, question: str) -> str:
        """Build system prompt with context."""
        return self.context_template.format(context=context, question=question)

    def query(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        preprocess: bool = True,
        include_context: bool = True,
        system_prompt_override: Optional[str] = None,
        **llm_kwargs,
    ) -> RAGResponse:
        """Execute full RAG pipeline.

        Args:
            question: User question
            top_k: Number of documents to retrieve (default: service setting)
            preprocess: Whether to preprocess query (default: True)
            include_context: Whether to include retrieved context (default: True)
            system_prompt_override: Override system prompt completely
            **llm_kwargs: Additional kwargs for LLM generation

        Returns:
            RAGResponse with answer and context
        """
        # 1. Preprocess query
        processed_query = self._preprocess_query(question) if preprocess else question

        # 2. Retrieve relevant documents
        k = top_k or self.retrieval_top_k
        results = self.search_service.search(processed_query, top_k=k)

        # 3. Build context
        if include_context and results:
            context = self._build_context(results)
            if system_prompt_override:
                system_prompt = system_prompt_override
            else:
                system_prompt = self._build_system_prompt(context, question)
        else:
            system_prompt = system_prompt_override

        # 4. Generate answer
        llm_response = self.chat_service.chat(
            question,
            system_prompt=system_prompt,
            **llm_kwargs,
        )

        return RAGResponse(
            answer=llm_response.text,
            context=results,
            question=question,
            metadata={
                "preprocessed_query": processed_query,
                "retrieval_top_k": k,
                "num_results": len(results),
            },
        )

    def query_simple(self, question: str, top_k: Optional[int] = None) -> str:
        """Simple query interface returning just the answer string.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            Answer string
        """
        response = self.query(question, top_k=top_k)
        return response.answer


__all__ = ["RAGService", "RAGResponse"]
