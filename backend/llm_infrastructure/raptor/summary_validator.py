"""NLI-based summary validation for RAPTOR.

This module implements summary quality validation:
1. Check if summary sentences are entailed by source chunks
2. Build evidence links (sentence -> supporting source IDs)
3. Filter out unsupported sentences
4. Compute overall validation score

Reference: SummaC (TACL 2022) - NLI-based factual consistency
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from backend.llm_infrastructure.raptor.schemas import ValidationResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class SentenceEvidence:
    """Evidence mapping for a single sentence.

    Attributes:
        sentence: The sentence text
        sentence_idx: Index in the summary
        is_supported: Whether sentence is supported by evidence
        support_score: Entailment score (0-1)
        evidence_ids: IDs of supporting source chunks
        evidence_scores: Score per evidence chunk
    """

    sentence: str
    sentence_idx: int
    is_supported: bool = False
    support_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    evidence_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidatorConfig:
    """Configuration for summary validator.

    Attributes:
        entailment_model: Model name for NLI (e.g., "cross-encoder/nli-deberta-v3-base")
        support_threshold: Minimum score to consider supported
        min_support_ratio: Minimum ratio of supported sentences for valid summary
        max_sentence_length: Maximum sentence length to process
        batch_size: Batch size for NLI inference
        use_embeddings_fallback: Use embedding similarity if NLI unavailable
    """

    entailment_model: str = "cross-encoder/nli-deberta-v3-base"
    support_threshold: float = 0.5
    min_support_ratio: float = 0.7
    max_sentence_length: int = 512
    batch_size: int = 32
    use_embeddings_fallback: bool = True


class SummaryValidator:
    """Validates summaries using NLI-based entailment checking.

    Ensures summary nodes in RAPTOR tree are factually grounded in
    their source chunks, preventing error propagation.

    Args:
        config: Validator configuration
    """

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.config = config or ValidatorConfig()
        self._model: CrossEncoder | None = None
        self._sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z가-힣])")

    def _load_model(self) -> "CrossEncoder":
        """Lazy load the NLI model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.config.entailment_model,
                    max_length=self.config.max_sentence_length,
                )
                logger.info(f"Loaded NLI model: {self.config.entailment_model}")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}")
                raise

        return self._model

    def validate_summary(
        self,
        summary: str,
        children_texts: list[str],
        children_ids: list[str] | None = None,
    ) -> ValidationResult:
        """Validate summary against source chunks.

        Args:
            summary: Summary text to validate
            children_texts: Source chunk texts
            children_ids: Source chunk IDs (optional)

        Returns:
            ValidationResult with scores and evidence links
        """
        if not summary or not children_texts:
            return ValidationResult(
                score=0.0,
                supported_ratio=0.0,
                details={"error": "Empty summary or sources"},
            )

        # Split summary into sentences
        sentences = self._split_sentences(summary)

        if not sentences:
            return ValidationResult(
                score=0.0,
                supported_ratio=0.0,
                details={"error": "No sentences found"},
            )

        # Generate IDs if not provided
        if children_ids is None:
            children_ids = [f"child_{i}" for i in range(len(children_texts))]

        # Validate each sentence
        try:
            sentence_results = self._validate_sentences(
                sentences, children_texts, children_ids
            )
        except Exception as e:
            logger.warning(f"NLI validation failed, using fallback: {e}")
            if self.config.use_embeddings_fallback:
                sentence_results = self._fallback_validation(
                    sentences, children_texts, children_ids
                )
            else:
                raise

        # Aggregate results
        supported_count = sum(1 for s in sentence_results if s.is_supported)
        supported_ratio = supported_count / len(sentences)

        # Build evidence links
        evidence_links: dict[str, list[str]] = {}
        unsupported: list[str] = []

        for result in sentence_results:
            if result.is_supported:
                evidence_links[result.sentence] = result.evidence_ids
            else:
                unsupported.append(result.sentence)

        # Compute overall score
        avg_score = np.mean([s.support_score for s in sentence_results])
        final_score = float(avg_score * supported_ratio)

        return ValidationResult(
            score=final_score,
            supported_ratio=supported_ratio,
            unsupported_sentences=unsupported,
            evidence_links=evidence_links,
            details={
                "total_sentences": len(sentences),
                "supported_sentences": supported_count,
                "avg_support_score": float(avg_score),
                "sentence_details": [
                    {
                        "sentence": s.sentence[:100],
                        "is_supported": s.is_supported,
                        "score": s.support_score,
                        "evidence_count": len(s.evidence_ids),
                    }
                    for s in sentence_results
                ],
            },
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split by common sentence delimiters
        raw_sentences = self._sentence_pattern.split(text)

        # Clean and filter
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Skip very short fragments
                sentences.append(sent)

        return sentences

    def _validate_sentences(
        self,
        sentences: list[str],
        sources: list[str],
        source_ids: list[str],
    ) -> list[SentenceEvidence]:
        """Validate sentences using NLI model.

        Args:
            sentences: Sentences to validate
            sources: Source texts
            source_ids: Source IDs

        Returns:
            List of SentenceEvidence
        """
        model = self._load_model()
        results: list[SentenceEvidence] = []

        for idx, sentence in enumerate(sentences):
            # Create NLI pairs (premise=source, hypothesis=sentence)
            pairs = [(source, sentence) for source in sources]

            # Predict entailment
            # NLI models typically output: [contradiction, neutral, entailment]
            scores = model.predict(pairs, batch_size=self.config.batch_size)

            # Handle different output formats
            if isinstance(scores[0], (list, np.ndarray)):
                # Multi-class output: take entailment class
                entailment_scores = [s[2] if len(s) > 2 else s[-1] for s in scores]
            else:
                # Single score output
                entailment_scores = list(scores)

            # Find supporting sources
            evidence_scores: dict[str, float] = {}
            evidence_ids: list[str] = []

            for source_idx, score in enumerate(entailment_scores):
                if score >= self.config.support_threshold:
                    source_id = source_ids[source_idx]
                    evidence_ids.append(source_id)
                    evidence_scores[source_id] = float(score)

            # Compute overall support
            max_score = max(entailment_scores) if entailment_scores else 0.0
            is_supported = max_score >= self.config.support_threshold

            results.append(
                SentenceEvidence(
                    sentence=sentence,
                    sentence_idx=idx,
                    is_supported=is_supported,
                    support_score=float(max_score),
                    evidence_ids=evidence_ids,
                    evidence_scores=evidence_scores,
                )
            )

        return results

    def _fallback_validation(
        self,
        sentences: list[str],
        sources: list[str],
        source_ids: list[str],
    ) -> list[SentenceEvidence]:
        """Fallback validation using embedding similarity.

        Args:
            sentences: Sentences to validate
            sources: Source texts
            source_ids: Source IDs

        Returns:
            List of SentenceEvidence (approximate)
        """
        logger.info("Using embedding similarity fallback for validation")

        # Simple word overlap / n-gram similarity as fallback
        results: list[SentenceEvidence] = []

        for idx, sentence in enumerate(sentences):
            sentence_tokens = set(sentence.lower().split())
            evidence_scores: dict[str, float] = {}
            evidence_ids: list[str] = []

            for source_idx, source in enumerate(sources):
                source_tokens = set(source.lower().split())

                # Jaccard similarity
                if sentence_tokens and source_tokens:
                    intersection = len(sentence_tokens & source_tokens)
                    union = len(sentence_tokens | source_tokens)
                    score = intersection / union if union > 0 else 0.0
                else:
                    score = 0.0

                if score >= 0.2:  # Lower threshold for word overlap
                    source_id = source_ids[source_idx]
                    evidence_ids.append(source_id)
                    evidence_scores[source_id] = score

            max_score = max(evidence_scores.values()) if evidence_scores else 0.0
            is_supported = max_score >= 0.2

            results.append(
                SentenceEvidence(
                    sentence=sentence,
                    sentence_idx=idx,
                    is_supported=is_supported,
                    support_score=max_score,
                    evidence_ids=evidence_ids,
                    evidence_scores=evidence_scores,
                )
            )

        return results

    def filter_unsupported(
        self,
        summary: str,
        children_texts: list[str],
    ) -> str:
        """Remove unsupported sentences from summary.

        Args:
            summary: Original summary
            children_texts: Source texts

        Returns:
            Filtered summary with only supported sentences
        """
        result = self.validate_summary(summary, children_texts)

        if result.is_valid:
            return summary

        # Rebuild from supported sentences
        sentences = self._split_sentences(summary)
        supported = [
            sent for sent in sentences if sent not in result.unsupported_sentences
        ]

        if not supported:
            # All unsupported - return first source as fallback
            return children_texts[0][:500] if children_texts else ""

        return " ".join(supported)

    def compute_self_retrieval_score(
        self,
        summary: str,
        children_embeddings: list[list[float]],
        summary_embedding: list[float],
        top_k: int = 5,
        negative_embeddings: list[list[float]] | None = None,
    ) -> float:
        """Compute self-retrieval score.

        Measures how well the summary can retrieve its source children
        from a pool that includes both children and negative samples.

        Args:
            summary: Summary text (unused, kept for API compatibility)
            children_embeddings: Embeddings of child chunks (positives)
            summary_embedding: Embedding of summary
            top_k: Number of top results to consider
            negative_embeddings: Optional negative samples to mix in pool

        Returns:
            Self-retrieval score (0-1)
        """
        if not children_embeddings:
            return 0.0

        summary_vec = np.array(summary_embedding)
        children_vecs = np.array(children_embeddings)
        num_children = len(children_embeddings)

        # Build retrieval pool: children + negatives
        if negative_embeddings and len(negative_embeddings) > 0:
            negative_vecs = np.array(negative_embeddings)
            pool_vecs = np.vstack([children_vecs, negative_vecs])
            # Mark which indices are children (True) vs negatives (False)
            is_child = [True] * num_children + [False] * len(negative_embeddings)
        else:
            # No negatives: compute cohesion-based score instead
            # Measures how tightly children cluster around the summary
            pool_vecs = children_vecs
            is_child = [True] * num_children

        # Compute similarities
        norms = np.linalg.norm(pool_vecs, axis=1)
        summary_norm = np.linalg.norm(summary_vec)

        if summary_norm < 1e-8:
            return 0.0

        # Avoid division by zero for zero vectors
        norms = np.where(norms < 1e-8, 1.0, norms)
        similarities = np.dot(pool_vecs, summary_vec) / (norms * summary_norm)

        if negative_embeddings and len(negative_embeddings) > 0:
            # Standard self-retrieval: check if children are in top-k
            top_k_actual = min(top_k, len(pool_vecs))
            top_indices = np.argsort(similarities)[-top_k_actual:]

            # Count how many of top-k are actual children
            retrieved_children = sum(1 for i in top_indices if is_child[i])
            expected_children = min(num_children, top_k_actual)

            return retrieved_children / expected_children if expected_children > 0 else 0.0
        else:
            # No negatives: use cohesion-based score
            # Score based on average similarity and variance
            child_similarities = similarities[:num_children]
            mean_sim = float(np.mean(child_similarities))
            min_sim = float(np.min(child_similarities))

            # Good summary: high mean similarity, high minimum (all children are close)
            # Score: weighted combination of mean and min
            # Threshold at 0.5 similarity for "good" clustering
            cohesion_score = (mean_sim * 0.6 + min_sim * 0.4)

            # Normalize to 0-1 range (assuming similarities are cosine: -1 to 1)
            return max(0.0, min(1.0, (cohesion_score + 1.0) / 2.0))


class BatchSummaryValidator:
    """Batch validator for multiple summaries.

    Optimizes NLI inference by batching across summaries.

    Args:
        config: Validator configuration
    """

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.validator = SummaryValidator(config)

    def validate_batch(
        self,
        summaries: list[str],
        children_texts_list: list[list[str]],
        children_ids_list: list[list[str]] | None = None,
    ) -> list[ValidationResult]:
        """Validate multiple summaries.

        Args:
            summaries: List of summaries
            children_texts_list: List of source text lists
            children_ids_list: List of source ID lists

        Returns:
            List of ValidationResults
        """
        if children_ids_list is None:
            children_ids_list = [None] * len(summaries)  # type: ignore

        results = []
        for summary, children_texts, children_ids in zip(
            summaries, children_texts_list, children_ids_list
        ):
            result = self.validator.validate_summary(
                summary, children_texts, children_ids
            )
            results.append(result)

        return results

    def get_low_quality_summaries(
        self,
        results: list[ValidationResult],
        threshold: float = 0.5,
    ) -> list[int]:
        """Get indices of low-quality summaries.

        Args:
            results: Validation results
            threshold: Quality threshold

        Returns:
            Indices of summaries below threshold
        """
        return [i for i, r in enumerate(results) if r.score < threshold]


__all__ = [
    "SummaryValidator",
    "BatchSummaryValidator",
    "ValidatorConfig",
    "SentenceEvidence",
]
