"""Mixture-of-Experts query routing for RAPTOR retrieval.

This module implements the QueryRouter class that:
1. Routes queries to the most relevant partition(s)
2. Computes p(g|q) = softmax(β·sim(q,c_g) + α·meta_match)
3. Provides graceful degradation when confidence is low
4. Supports global fallback for uncertain queries

The routing is a Mixture-of-Experts style approach where each
partition is an "expert" and queries are routed based on
semantic + metadata matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from backend.llm_infrastructure.raptor.schemas import Partition

if TYPE_CHECKING:
    from backend.llm_infrastructure.embedding.base import BaseEmbedder
    from backend.llm_infrastructure.llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class QueryRouterConfig:
    """Configuration for query routing.

    Attributes:
        beta: Semantic similarity weight
        alpha_device: Device name match weight
        alpha_doc_type: Doc type match weight
        global_fallback_weight: Base weight for global search
        confidence_threshold: Threshold for high-confidence routing
        top_k_groups: Maximum groups to route to
        use_llm_extraction: Whether to use LLM for metadata extraction
        device_patterns: Regex patterns for device name extraction
        doc_type_patterns: Regex patterns for doc type extraction
    """

    beta: float = 1.0
    alpha_device: float = 0.8
    alpha_doc_type: float = 0.5
    global_fallback_weight: float = 0.1
    confidence_threshold: float = 0.6
    top_k_groups: int = 3
    use_llm_extraction: bool = False
    device_patterns: list[str] = field(default_factory=lambda: [
        r"(?:SUPRA|EFEM|RFID|EXP|SORTER)\s*(?:XP|[\w\d]+)?",
        r"장비[:\s]*([^\s,]+)",
    ])
    doc_type_patterns: list[str] = field(default_factory=lambda: [
        r"(?:SOP|TS|매뉴얼|가이드|절차서|trouble\s*shoot)",
        r"문서[:\s]*([^\s,]+)",
    ])


@dataclass
class RoutingDistribution:
    """Query routing result with probability distribution.

    Attributes:
        group_weights: Mapping of group_id to routing weight
        extracted_metadata: Metadata extracted from query
        confidence: Overall routing confidence
        top_group: Highest weight group
        is_high_confidence: Whether routing is confident
    """

    group_weights: dict[str, float]
    extracted_metadata: dict[str, str | None] = field(default_factory=dict)
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def top_group(self) -> str | None:
        """Get group with highest weight."""
        if not self.group_weights:
            return None
        return max(self.group_weights, key=self.group_weights.get)

    def is_high_confidence_at(self, threshold: float = 0.6) -> bool:
        """Check if routing is high confidence at given threshold."""
        return self.confidence >= threshold

    @property
    def is_high_confidence(self) -> bool:
        """Check if routing is high confidence (default threshold 0.6)."""
        return self.confidence >= 0.6

    def get_top_k(self, k: int) -> list[tuple[str, float]]:
        """Get top-k groups by weight."""
        sorted_groups = sorted(
            self.group_weights.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_groups[:k]


class QueryRouter:
    """Routes queries to relevant partitions using MoE approach.

    Computes routing distribution p(g|q) based on:
    - Semantic similarity between query and group centroids
    - Metadata match from extracted query metadata
    - Global fallback for uncertain queries

    Args:
        embedder: Embedder for query embedding
        partitions: List of available partitions
        config: Router configuration
        llm: Optional LLM for metadata extraction
    """

    def __init__(
        self,
        embedder: "BaseEmbedder",
        partitions: list[Partition] | None = None,
        config: QueryRouterConfig | None = None,
        llm: "BaseLLM | None" = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or QueryRouterConfig()
        self.llm = llm
        self._partitions: dict[str, Partition] = {}
        self._device_vocab: set[str] = set()
        self._doc_type_vocab: set[str] = set()

        # Compile regex patterns
        self._device_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.device_patterns
        ]
        self._doc_type_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.doc_type_patterns
        ]

        if partitions:
            self.set_partitions(partitions)

    def set_partitions(self, partitions: list[Partition]) -> None:
        """Set available partitions and build vocabulary."""
        self._partitions = {p.key: p for p in partitions}

        # Build vocabulary from partitions
        for p in partitions:
            if p.device_name:
                self._device_vocab.add(p.device_name.upper())
            if p.doc_type:
                self._doc_type_vocab.add(p.doc_type.lower())

        logger.info(
            f"Set {len(partitions)} partitions, "
            f"devices: {self._device_vocab}, doc_types: {self._doc_type_vocab}"
        )

    def route(self, query: str) -> RoutingDistribution:
        """Route a query to relevant groups.

        Args:
            query: User query text

        Returns:
            RoutingDistribution with group weights
        """
        if not self._partitions:
            return RoutingDistribution(
                group_weights={"global": 1.0},
                confidence=0.0,
                details={"error": "No partitions available"},
            )

        # Extract metadata from query
        extracted = self._extract_metadata(query)

        # Embed query
        query_embedding = self.embedder.embed(query)

        # Compute scores for all groups
        scores = self._compute_group_scores(query_embedding, extracted)

        # Normalize to weights
        weights = self._normalize_to_weights(scores)

        # Add global fallback
        weights = self._add_global_fallback(weights, extracted)

        # Compute confidence
        confidence = self._compute_confidence(weights)

        return RoutingDistribution(
            group_weights=weights,
            extracted_metadata=extracted,
            confidence=confidence,
            details={
                "raw_scores": scores,
                "extracted_device": extracted.get("device_name"),
                "extracted_doc_type": extracted.get("doc_type"),
            },
        )

    def route_filtered(self, query: str) -> RoutingDistribution:
        """Route a query and filter based on config thresholds.

        Uses config.confidence_threshold and config.top_k_groups to filter results.

        Args:
            query: User query text

        Returns:
            RoutingDistribution with filtered group weights
        """
        result = self.route(query)

        # Filter to top_k groups (excluding global)
        non_global = {k: v for k, v in result.group_weights.items() if k != "global"}
        sorted_groups = sorted(non_global.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_groups[: self.config.top_k_groups]

        # Keep global if present
        filtered_weights = dict(top_k)
        if "global" in result.group_weights:
            filtered_weights["global"] = result.group_weights["global"]

        # Renormalize non-global weights
        non_global_sum = sum(v for k, v in filtered_weights.items() if k != "global")
        global_weight = filtered_weights.get("global", 0.0)

        if non_global_sum > 0:
            target_sum = 1.0 - global_weight
            scale = target_sum / non_global_sum
            filtered_weights = {
                k: v * scale if k != "global" else v
                for k, v in filtered_weights.items()
            }

        return RoutingDistribution(
            group_weights=filtered_weights,
            extracted_metadata=result.extracted_metadata,
            confidence=result.confidence,
            details={
                **result.details,
                "is_high_confidence": result.is_high_confidence_at(
                    self.config.confidence_threshold
                ),
                "top_k_groups": self.config.top_k_groups,
            },
        )

    def _extract_metadata(self, query: str) -> dict[str, str | None]:
        """Extract metadata from query using rules and optionally LLM.

        Args:
            query: Query text

        Returns:
            Dictionary with extracted metadata
        """
        result: dict[str, str | None] = {
            "device_name": None,
            "doc_type": None,
        }

        # Rule-based extraction
        query_upper = query.upper()
        query_lower = query.lower()

        # Device extraction
        for device in self._device_vocab:
            if device in query_upper:
                result["device_name"] = device
                break

        if not result["device_name"]:
            for pattern in self._device_patterns:
                match = pattern.search(query)
                if match:
                    extracted = match.group(0).strip()
                    # Normalize
                    for device in self._device_vocab:
                        if extracted.upper() in device or device in extracted.upper():
                            result["device_name"] = device
                            break
                    break

        # Doc type extraction
        for doc_type in self._doc_type_vocab:
            if doc_type in query_lower:
                result["doc_type"] = doc_type
                break

        if not result["doc_type"]:
            for pattern in self._doc_type_patterns:
                match = pattern.search(query)
                if match:
                    extracted = match.group(0).strip().lower()
                    # Map common terms
                    if "sop" in extracted or "절차" in extracted:
                        result["doc_type"] = "sop"
                    elif "ts" in extracted or "trouble" in extracted:
                        result["doc_type"] = "ts"
                    elif "매뉴얼" in extracted or "manual" in extracted:
                        result["doc_type"] = "manual"
                    elif "가이드" in extracted or "guide" in extracted:
                        result["doc_type"] = "guide"
                    break

        # LLM extraction if enabled and rules failed
        if self.config.use_llm_extraction and self.llm:
            if not result["device_name"] or not result["doc_type"]:
                llm_result = self._llm_extract_metadata(query)
                if not result["device_name"] and llm_result.get("device_name"):
                    result["device_name"] = llm_result["device_name"]
                if not result["doc_type"] and llm_result.get("doc_type"):
                    result["doc_type"] = llm_result["doc_type"]

        return result

    def _llm_extract_metadata(self, query: str) -> dict[str, str | None]:
        """Extract metadata using LLM.

        Args:
            query: Query text

        Returns:
            Extracted metadata
        """
        if not self.llm:
            return {"device_name": None, "doc_type": None}

        prompt = f"""Extract equipment name and document type from this query.

Query: {query}

Available equipment: {', '.join(self._device_vocab)}
Available document types: {', '.join(self._doc_type_vocab)}

Return in format:
equipment: <name or None>
doc_type: <type or None>"""

        try:
            response = self.llm.generate(prompt, max_tokens=50)
            # Parse response
            result: dict[str, str | None] = {"device_name": None, "doc_type": None}
            for line in response.split("\n"):
                if "equipment:" in line.lower():
                    val = line.split(":", 1)[1].strip()
                    if val.lower() != "none":
                        result["device_name"] = val.upper()
                elif "doc_type:" in line.lower():
                    val = line.split(":", 1)[1].strip()
                    if val.lower() != "none":
                        result["doc_type"] = val.lower()
            return result
        except Exception as e:
            logger.warning(f"LLM metadata extraction failed: {e}")
            return {"device_name": None, "doc_type": None}

    def _compute_group_scores(
        self,
        query_embedding: list[float],
        extracted: dict[str, str | None],
    ) -> dict[str, float]:
        """Compute routing scores for all groups.

        score(g) = β·sim(q, c_g) + α_d·match_device + α_t·match_type

        Args:
            query_embedding: Query embedding vector
            extracted: Extracted metadata

        Returns:
            Dictionary of group_id to score
        """
        scores: dict[str, float] = {}
        query_vec = np.array(query_embedding)

        for key, partition in self._partitions.items():
            if partition.stats is None or not partition.stats.centroid:
                continue

            score = 0.0

            # Semantic similarity
            centroid = np.array(partition.stats.centroid)
            similarity = self._cosine_similarity(query_vec, centroid)
            score += self.config.beta * similarity

            # Metadata match
            if extracted.get("device_name") and partition.device_name:
                if extracted["device_name"].upper() == partition.device_name.upper():
                    score += self.config.alpha_device
            if extracted.get("doc_type") and partition.doc_type:
                if extracted["doc_type"].lower() == partition.doc_type.lower():
                    score += self.config.alpha_doc_type

            scores[key] = score

        return scores

    def _normalize_to_weights(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Normalize scores to weights using softmax.

        Args:
            scores: Raw scores

        Returns:
            Normalized weights summing to 1
        """
        if not scores:
            return {}

        score_values = np.array(list(scores.values()))
        score_keys = list(scores.keys())

        # Softmax
        exp_scores = np.exp(score_values - np.max(score_values))
        softmax_weights = exp_scores / (np.sum(exp_scores) + 1e-8)

        return {k: float(w) for k, w in zip(score_keys, softmax_weights)}

    def _add_global_fallback(
        self,
        weights: dict[str, float],
        extracted: dict[str, str | None],
    ) -> dict[str, float]:
        """Add global fallback weight for safety.

        Higher fallback when metadata not extracted.

        Args:
            weights: Current weights
            extracted: Extracted metadata

        Returns:
            Updated weights with global fallback
        """
        # Compute fallback weight based on extraction success
        fallback_weight = self.config.global_fallback_weight

        if not extracted.get("device_name") and not extracted.get("doc_type"):
            # No metadata extracted - higher fallback
            fallback_weight = min(0.3, fallback_weight * 3)
        elif not extracted.get("device_name") or not extracted.get("doc_type"):
            # Partial metadata
            fallback_weight = min(0.2, fallback_weight * 2)

        # Redistribute weights
        if weights and fallback_weight > 0:
            scale = 1.0 - fallback_weight
            weights = {k: v * scale for k, v in weights.items()}
            weights["global"] = fallback_weight
        elif not weights:
            weights = {"global": 1.0}

        return weights

    def _compute_confidence(self, weights: dict[str, float]) -> float:
        """Compute routing confidence.

        High confidence when one group dominates.

        Args:
            weights: Normalized weights

        Returns:
            Confidence score (0-1)
        """
        if not weights:
            return 0.0

        non_global_weights = [w for k, w in weights.items() if k != "global"]

        if not non_global_weights:
            return 0.0

        # Confidence based on max weight and entropy
        max_weight = max(non_global_weights)

        # Handle edge case: single group
        if len(non_global_weights) == 1:
            # Single group: confidence is based on how much weight goes to it vs global
            global_weight = weights.get("global", 0.0)
            # High confidence if global fallback is low
            return float(max_weight * (1.0 - global_weight))

        # Multiple groups: use entropy-based confidence
        # Normalize weights for entropy calculation
        total_weight = sum(non_global_weights)
        if total_weight < 1e-8:
            return 0.0

        normalized = [w / total_weight for w in non_global_weights]
        entropy = -sum(w * np.log(w + 1e-8) for w in normalized if w > 0)
        max_entropy = np.log(len(non_global_weights))  # log(n) for n groups

        # Lower entropy = higher confidence
        entropy_factor = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        return float(max_weight * 0.5 + entropy_factor * 0.5)

    def _cosine_similarity(
        self,
        vec1: NDArray[np.floating[Any]],
        vec2: NDArray[np.floating[Any]],
    ) -> float:
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return {
            "num_partitions": len(self._partitions),
            "device_vocab": list(self._device_vocab),
            "doc_type_vocab": list(self._doc_type_vocab),
            "config": {
                "beta": self.config.beta,
                "alpha_device": self.config.alpha_device,
                "alpha_doc_type": self.config.alpha_doc_type,
                "global_fallback_weight": self.config.global_fallback_weight,
            },
        }


class AdaptiveQueryRouter(QueryRouter):
    """Query router with adaptive weight tuning.

    Learns weights from feedback signals:
    - User clicks / relevance feedback
    - Retrieval success/failure

    Args:
        **kwargs: Arguments passed to QueryRouter
        learning_rate: Weight update rate
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self._feedback_history: list[dict[str, Any]] = []

    def record_feedback(
        self,
        query: str,
        routing_result: RoutingDistribution,
        actual_group: str,
        success: bool,
    ) -> None:
        """Record routing feedback for learning.

        Args:
            query: Original query
            routing_result: Routing that was used
            actual_group: Group that was actually relevant
            success: Whether retrieval was successful
        """
        self._feedback_history.append({
            "query": query,
            "predicted_top": routing_result.top_group,
            "actual_group": actual_group,
            "success": success,
            "extracted_meta": routing_result.extracted_metadata,
        })

        # Simple weight adjustment
        if not success and routing_result.top_group != actual_group:
            # Increase beta if semantic was wrong
            if routing_result.extracted_metadata.get("device_name"):
                # Meta was available but routing still wrong
                self.config.alpha_device += self.learning_rate
            else:
                self.config.beta -= self.learning_rate * 0.5

        # Clamp weights
        self.config.beta = max(0.1, min(3.0, self.config.beta))
        self.config.alpha_device = max(0.1, min(2.0, self.config.alpha_device))

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get feedback statistics."""
        if not self._feedback_history:
            return {"total_feedback": 0}

        successes = sum(1 for f in self._feedback_history if f["success"])
        matches = sum(
            1 for f in self._feedback_history
            if f["predicted_top"] == f["actual_group"]
        )

        return {
            "total_feedback": len(self._feedback_history),
            "success_rate": successes / len(self._feedback_history),
            "routing_accuracy": matches / len(self._feedback_history),
            "current_beta": self.config.beta,
            "current_alpha_device": self.config.alpha_device,
        }


__all__ = [
    "QueryRouter",
    "AdaptiveQueryRouter",
    "QueryRouterConfig",
    "RoutingDistribution",
]
