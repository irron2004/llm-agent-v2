"""Stratified pooling strategy ensuring doc_type diversity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import PoolingStrategy
from scripts.golden_set.config import PooledDocument, PoolingConfig

if TYPE_CHECKING:
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
    from backend.llm_infrastructure.embedding.base import BaseEmbedder


class StratifiedPooler(PoolingStrategy):
    """Stratified pooling ensuring minimum documents per doc_type."""

    def __init__(
        self,
        es_engine: "EsSearchEngine",
        config: PoolingConfig,
        embedder: "BaseEmbedder",
    ) -> None:
        super().__init__(es_engine, config)
        self.embedder = embedder

    @property
    def method_name(self) -> str:
        return "stratified"

    def pool(self, query: str, top_k: int) -> list[PooledDocument]:
        """Retrieve documents ensuring doc_type diversity.

        Strategy (Min Quota):
        1. Perform hybrid search for top_k documents
        2. Count documents per doc_type
        3. For underrepresented types, perform filtered search to meet minimum

        Args:
            query: Search query text.
            top_k: Number of documents to retrieve.

        Returns:
            List of PooledDocument objects.
        """
        query_vec = self._embed_query(query)

        # Step 1: Get initial results from hybrid search
        all_hits = self.es_engine.hybrid_search(
            query_vector=query_vec,
            query_text=query,
            top_k=top_k,
            use_rrf=True,
        )
        all_docs = [self._hit_to_pooled_doc(h, self.method_name) for h in all_hits]

        # Step 2: Count by doc_type
        type_counts: dict[str, int] = {}
        for doc in all_docs:
            dt = doc.doc_type
            type_counts[dt] = type_counts.get(dt, 0) + 1

        # Step 3: Find underrepresented types and add more
        existing_ids = {d.chunk_id for d in all_docs}
        additional: list[PooledDocument] = []

        for doc_type in self.config.doc_types:
            current_count = type_counts.get(doc_type, 0)
            if current_count < self.config.min_per_doc_type:
                needed = self.config.min_per_doc_type - current_count

                # Build filter for this doc_type
                filter_clause = self.es_engine.build_filter(doc_type=doc_type)

                # Search with filter
                extra_hits = self.es_engine.hybrid_search(
                    query_vector=query_vec,
                    query_text=query,
                    top_k=needed + 5,  # Get a few extra
                    filters=filter_clause,
                    use_rrf=True,
                )

                added = 0
                for hit in extra_hits:
                    if hit.chunk_id not in existing_ids:
                        doc = self._hit_to_pooled_doc(hit, self.method_name)
                        additional.append(doc)
                        existing_ids.add(hit.chunk_id)
                        added += 1
                        if added >= needed:
                            break

        return all_docs + additional

    def _embed_query(self, query: str) -> list[float]:
        """Embed query and L2 normalize."""
        vec = self.embedder.embed_batch([query])[0]
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


__all__ = ["StratifiedPooler"]
