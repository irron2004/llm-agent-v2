"""Section Expander: 검색 결과의 section/chapter 그룹을 확장하는 후처리기.

검색+리랭크 후 top-k 결과에서 section_chapter 기반으로
같은 섹션의 모든 페이지를 가져와 컨텍스트를 확장한다.

Usage:
    expander = SectionExpander.from_settings(rag_settings)
    result = expander.expand(search_hits, es_engine)
    final_hits = result.all_results_ordered()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.config.settings import RAGSettings
    from backend.llm_infrastructure.retrieval.engines.es_search import (
        EsSearchEngine,
        EsSearchHit,
    )

logger = logging.getLogger(__name__)


@dataclass
class ExpandedGroup:
    """하나의 확장된 섹션 그룹."""

    trigger_hit: "EsSearchHit"
    section_chunks: list["EsSearchHit"]
    doc_id: str
    section_chapter: str

    @property
    def all_chunks(self) -> list["EsSearchHit"]:
        """트리거 히트 포함 전체 청크 (페이지순)."""
        seen = {self.trigger_hit.chunk_id}
        result = []
        for chunk in self.section_chunks:
            if chunk.chunk_id not in seen:
                result.append(chunk)
                seen.add(chunk.chunk_id)
        # Insert trigger hit at correct page position
        trigger_page = self.trigger_hit.page or 0
        inserted = False
        final = []
        for chunk in result:
            chunk_page = chunk.page or 0
            if not inserted and chunk_page >= trigger_page:
                final.append(self.trigger_hit)
                inserted = True
            final.append(chunk)
        if not inserted:
            final.append(self.trigger_hit)
        return final


@dataclass
class SectionExpandResult:
    """섹션 확장 결과."""

    original_hits: list["EsSearchHit"]
    expanded_groups: list[ExpandedGroup]
    unexpanded_hits: list["EsSearchHit"]

    def all_results_ordered(self) -> list["EsSearchHit"]:
        """원래 순위 유지하면서 확장된 청크를 삽입한 최종 결과.

        확장된 그룹의 트리거 히트 위치에 섹션 전체 청크를 삽입하고,
        나머지 히트는 원래 순서 유지.
        """
        # Build expansion map: chunk_id -> ExpandedGroup
        expansion_map: dict[str, ExpandedGroup] = {}
        for group in self.expanded_groups:
            expansion_map[group.trigger_hit.chunk_id] = group

        result: list["EsSearchHit"] = []
        seen_chunk_ids: set[str] = set()

        for hit in self.original_hits:
            if hit.chunk_id in seen_chunk_ids:
                continue

            if hit.chunk_id in expansion_map:
                group = expansion_map[hit.chunk_id]
                for chunk in group.all_chunks:
                    if chunk.chunk_id not in seen_chunk_ids:
                        result.append(chunk)
                        seen_chunk_ids.add(chunk.chunk_id)
            else:
                if hit.chunk_id not in seen_chunk_ids:
                    result.append(hit)
                    seen_chunk_ids.add(hit.chunk_id)

        return result


class SectionExpander:
    """검색 결과에서 section_chapter 기반으로 섹션을 확장하는 후처리기."""

    def __init__(
        self,
        enabled: bool = True,
        top_groups: int = 2,
        max_pages: int = 8,
        allowed_sources: set[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.top_groups = top_groups
        self.max_pages = max_pages
        self.allowed_sources = allowed_sources or {"title", "rule", "toc_match", "carry"}

    @classmethod
    def from_settings(cls, settings: "RAGSettings") -> "SectionExpander":
        """RAGSettings에서 SectionExpander 생성."""
        allowed = set(
            s.strip()
            for s in settings.section_expand_allowed_sources.split(",")
            if s.strip()
        )
        return cls(
            enabled=settings.section_expand_enabled,
            top_groups=settings.section_expand_top_groups,
            max_pages=settings.section_expand_max_pages,
            allowed_sources=allowed,
        )

    def expand(
        self,
        hits: list["EsSearchHit"],
        es_engine: "EsSearchEngine",
        content_index: str | None = None,
    ) -> SectionExpandResult:
        """검색 결과에서 상위 그룹의 섹션을 확장.

        Args:
            hits: 검색+리랭크 후 결과
            es_engine: ES 검색 엔진 (fetch_section_chunks 호출용)
            content_index: content 인덱스명 (None이면 es_engine 기본값)

        Returns:
            SectionExpandResult
        """
        if not self.enabled or not hits:
            return SectionExpandResult(
                original_hits=hits,
                expanded_groups=[],
                unexpanded_hits=hits,
            )

        # Identify unique (doc_id, section_chapter) groups from top hits
        seen_groups: set[tuple[str, str]] = set()
        candidate_hits: list["EsSearchHit"] = []

        for hit in hits:
            meta = hit.metadata or {}
            section_chapter = str(meta.get("section_chapter", "") or "")
            chapter_source = str(meta.get("chapter_source", "") or "")
            chapter_ok = bool(meta.get("chapter_ok", False))

            if (
                not section_chapter
                or chapter_source not in self.allowed_sources
            ):
                continue

            group_key = (hit.doc_id, section_chapter)
            if group_key in seen_groups:
                continue

            seen_groups.add(group_key)
            candidate_hits.append(hit)

            if chapter_source == "carry":
                logger.warning(
                    "[SectionExpander] CARRY-TRIGGERED expansion: "
                    "doc_id=%s, page=%s, chapter='%s', chapter_ok=%s, source=%s",
                    hit.doc_id, hit.page, section_chapter, chapter_ok, chapter_source,
                )
            else:
                logger.info(
                    "[SectionExpander] expansion trigger: "
                    "doc_id=%s, page=%s, chapter='%s', source=%s",
                    hit.doc_id, hit.page, section_chapter, chapter_source,
                )

            if len(candidate_hits) >= self.top_groups:
                break

        # Expand each group
        expanded_groups: list[ExpandedGroup] = []
        expanded_chunk_ids: set[str] = set()

        for hit in candidate_hits:
            meta = hit.metadata or {}
            section_chapter = str(meta.get("section_chapter", "") or "")

            section_chunks = es_engine.fetch_section_chunks(
                doc_id=hit.doc_id,
                section_chapter=section_chapter,
                max_pages=self.max_pages,
                content_index=content_index,
            )

            if section_chunks:
                group = ExpandedGroup(
                    trigger_hit=hit,
                    section_chunks=section_chunks,
                    doc_id=hit.doc_id,
                    section_chapter=section_chapter,
                )
                expanded_groups.append(group)
                for chunk in group.all_chunks:
                    expanded_chunk_ids.add(chunk.chunk_id)

                chapter_source = str((hit.metadata or {}).get("chapter_source", ""))
                pages = sorted({c.page for c in group.all_chunks if c.page})
                logger.info(
                    "[SectionExpander] expanded group: doc_id=%s, chapter='%s', "
                    "source=%s, pages=%s (%d chunks)",
                    hit.doc_id, section_chapter, chapter_source,
                    f"{pages[0]}-{pages[-1]}" if pages else "?",
                    len(group.all_chunks),
                )

        # Hits not part of any expanded group
        unexpanded = [
            h for h in hits if h.chunk_id not in expanded_chunk_ids
        ]

        return SectionExpandResult(
            original_hits=hits,
            expanded_groups=expanded_groups,
            unexpanded_hits=unexpanded,
        )


__all__ = ["SectionExpander", "SectionExpandResult", "ExpandedGroup"]
