"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from backend.llm_infrastructure.retrieval.base import RetrievalResult
from ..base import BaseReranker
from ..registry import register_reranker

logger = logging.getLogger(__name__)


@register_reranker("cross_encoder", version="v1")
class CrossEncoderReranker(BaseReranker):
    """Reranker using Cross-Encoder models from sentence-transformers.

    Cross-encoders jointly encode query-document pairs and produce
    a relevance score, providing more accurate ranking than bi-encoders
    at the cost of computational efficiency.

    Common models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
        - BAAI/bge-reranker-base (multilingual)
        - BAAI/bge-reranker-v2-m3 (multilingual, high quality)
    """

    # Default model for cross-encoder reranking
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name/path (default: ms-marco-MiniLM-L-6-v2)
            device: Device to run model on (cpu/cuda/auto)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Lazy loading - model loaded on first use
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        """Load the cross-encoder model (lazy loading)."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            logger.info(f"Cross-encoder model loaded on device: {self.device}")
        return self._model

    @property
    def model(self) -> CrossEncoder:
        """Get the cross-encoder model (loads if not already loaded)."""
        return self._load_model()

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using cross-encoder.

        Args:
            query: Original search query
            results: List of retrieval results to rerank
            top_k: Number of results to return (None = return all reranked)
            **kwargs: Additional parameters (unused)

        Returns:
            Reranked results sorted by cross-encoder score
        """
        if not results:
            return []

        # Prepare query-document pairs
        # Use raw_text if available (original content), otherwise use content
        pairs = [
            (query, r.raw_text if r.raw_text else r.content)
            for r in results
        ]

        # Get cross-encoder scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Create reranked results with new scores
        reranked: list[RetrievalResult] = []
        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    score=float(score),
                    metadata={
                        **(result.metadata or {}),
                        "original_score": result.score,
                        "rerank_model": self.model_name,
                    },
                    raw_text=result.raw_text,
                )
            )

        # Sort by new score (descending)
        reranked.sort(key=lambda r: r.score, reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model={self.model_name}, "
            f"device={self.device})"
        )


__all__ = ["CrossEncoderReranker"]