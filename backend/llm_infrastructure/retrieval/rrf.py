from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .base import RetrievalResult


DedupeKey: TypeAlias = tuple[str] | tuple[str, str]


@dataclass
class _MergedCandidate:
    rrf_score: float
    best_rank: int
    representative: RetrievalResult


def _metadata_value(result: RetrievalResult, key: str) -> object | None:
    metadata = result.metadata or {}
    return metadata.get(key)


def _has_value(value: object | None) -> bool:
    return value not in (None, "")


def _dedupe_key(result: RetrievalResult) -> DedupeKey:
    chunk_id = _metadata_value(result, "chunk_id")
    if _has_value(chunk_id):
        return (result.doc_id, str(chunk_id))

    page = _metadata_value(result, "page")
    if _has_value(page):
        return (result.doc_id, str(page))

    return (result.doc_id,)


def _sortable_intish(value: object | None) -> tuple[int, int, str]:
    if _has_value(value):
        try:
            return (0, int(str(value)), "")
        except (TypeError, ValueError):
            return (1, 0, str(value))
    return (2, 0, "")


def merge_retrieval_results_rrf(
    stage1: list[RetrievalResult],
    stage2: list[RetrievalResult],
    *,
    k: int = 60,
) -> list[RetrievalResult]:
    return merge_retrieval_result_lists_rrf([stage1, stage2], k=k)


def merge_retrieval_result_lists_rrf(
    result_lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
) -> list[RetrievalResult]:
    if k < 0:
        raise ValueError("k must be >= 0")

    merged: dict[DedupeKey, _MergedCandidate] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = _dedupe_key(result)
            contribution = 1.0 / (k + rank)
            candidate = merged.get(key)

            if candidate is None:
                merged[key] = _MergedCandidate(
                    rrf_score=contribution,
                    best_rank=rank,
                    representative=result,
                )
                continue

            candidate.rrf_score += contribution
            if rank < candidate.best_rank:
                candidate.best_rank = rank
                candidate.representative = result

    output: list[tuple[RetrievalResult, float, int]] = []
    for candidate in merged.values():
        base = candidate.representative
        output.append(
            (
                RetrievalResult(
                    doc_id=base.doc_id,
                    content=base.content,
                    score=candidate.rrf_score,
                    metadata=base.metadata,
                    raw_text=base.raw_text,
                ),
                candidate.rrf_score,
                candidate.best_rank,
            )
        )

    output.sort(
        key=lambda item: (
            -item[1],
            item[2],
            item[0].doc_id,
            _sortable_intish(_metadata_value(item[0], "page")),
            _sortable_intish(_metadata_value(item[0], "chunk_id")),
        )
    )

    return [result for result, _, _ in output]


__all__ = [
    "merge_retrieval_result_lists_rrf",
    "merge_retrieval_results_rrf",
]
