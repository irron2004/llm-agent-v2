"""Near-duplicate removal for pooled documents."""

from __future__ import annotations

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.golden_set.config import PooledDocument, PoolingConfig


def deduplicate_pool(
    documents: list[PooledDocument],
    config: PoolingConfig,
) -> list[PooledDocument]:
    """Remove near-duplicate documents from pool.

    Uses TF-IDF based cosine similarity with doc_type-specific thresholds.
    For each cluster of similar documents, keeps the one with highest score.

    Args:
        documents: List of documents to deduplicate.
        config: Pooling configuration with thresholds.

    Returns:
        Deduplicated list of documents.
    """
    if len(documents) <= 1:
        return documents

    # Group by doc_type
    by_type: dict[str, list[PooledDocument]] = defaultdict(list)
    for doc in documents:
        by_type[doc.doc_type].append(doc)

    deduplicated: list[PooledDocument] = []

    for doc_type, docs in by_type.items():
        if len(docs) <= 1:
            deduplicated.extend(docs)
            continue

        # TF-IDF vectorization
        texts = [d.content for d in docs]
        vectorizer = TfidfVectorizer(max_features=5000)

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Texts too short or empty
            deduplicated.extend(docs)
            continue

        # Compute similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Get threshold for this doc_type
        threshold = config.doc_type_thresholds.get(
            doc_type, config.similarity_threshold
        )

        # Greedy clustering
        used: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(len(docs)):
            if i in used:
                continue

            cluster = [i]
            for j in range(i + 1, len(docs)):
                if j not in used and sim_matrix[i, j] >= threshold:
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        # Select representative from each cluster (highest score)
        for cluster in clusters:
            cluster_docs = [(docs[idx], docs[idx].score) for idx in cluster]
            cluster_docs.sort(key=lambda x: x[1], reverse=True)
            representative = cluster_docs[0][0]
            deduplicated.append(representative)

    return deduplicated


def merge_and_deduplicate(
    pool_lists: list[list[PooledDocument]],
    config: PoolingConfig,
) -> list[PooledDocument]:
    """Merge multiple pool results and remove duplicates.

    Steps:
    1. Remove exact duplicates (same chunk_id)
    2. Remove near-duplicates (high TF-IDF similarity)
    3. Limit to total_pool_size

    Args:
        pool_lists: List of document lists from different poolers.
        config: Pooling configuration.

    Returns:
        Merged and deduplicated documents.
    """
    # Step 1: Remove exact duplicates (keep first occurrence)
    seen_ids: set[str] = set()
    merged: list[PooledDocument] = []

    for pool in pool_lists:
        for doc in pool:
            if doc.chunk_id not in seen_ids:
                seen_ids.add(doc.chunk_id)
                merged.append(doc)

    # Step 2: Remove near-duplicates
    deduplicated = deduplicate_pool(merged, config)

    # Step 3: Limit pool size (sort by score, take top N)
    if len(deduplicated) > config.total_pool_size:
        deduplicated.sort(key=lambda d: d.score, reverse=True)
        deduplicated = deduplicated[: config.total_pool_size]

    return deduplicated


__all__ = ["deduplicate_pool", "merge_and_deduplicate"]
