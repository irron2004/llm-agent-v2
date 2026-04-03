"""Relation Expander: 검색 결과에서 관계 기반으로 관련 chunk를 확장하는 후처리기.

검색+리랭크 후 top-k 결과에서 다음 4가지 전략으로 관련 chunk를 가져온다:
1. cross_doc_type: 같은 장비 + 다른 문서 타입 (SOP→TS, SOP→MyService 등)
2. equip_history: 같은 equip_id 과거 이력 (MyService)
3. component_link: 같은 components 필드를 가진 다른 문서의 chunk
4. sequential: 같은 doc_id 내 앞뒤 페이지 chunk

Usage:
    expander = RelationExpander.from_settings(rag_settings)
    result = expander.expand(search_hits, es_client, content_index)
    expanded = result.all_results()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.config.settings import RAGSettings
    from backend.llm_infrastructure.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RelationGroup:
    """하나의 관계 확장 그룹."""

    trigger_doc_id: str
    trigger_chunk_id: str
    related_results: list["RetrievalResult"]
    relation_type: str  # cross_doc_type, equip_history, component_link, sequential


@dataclass
class RelationExpandResult:
    """관계 확장 결과."""

    original_results: list["RetrievalResult"]
    expanded_groups: list[RelationGroup]

    def all_results(self) -> list["RetrievalResult"]:
        """원래 결과 + 확장된 결과를 합친 최종 리스트 (중복 제거)."""
        seen_chunk_ids: set[str] = set()
        result: list["RetrievalResult"] = []

        # Original results first
        for r in self.original_results:
            cid = (r.metadata or {}).get("chunk_id", r.doc_id)
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                result.append(r)

        # Then expanded results
        for group in self.expanded_groups:
            for r in group.related_results:
                cid = (r.metadata or {}).get("chunk_id", r.doc_id)
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    # Tag with relation type
                    metadata = dict(r.metadata or {})
                    metadata["relation_type"] = group.relation_type
                    metadata["relation_trigger"] = group.trigger_chunk_id
                    r.metadata = metadata
                    result.append(r)

        return result


class RelationExpander:
    """검색 결과에서 관계 기반으로 관련 chunk를 확장하는 후처리기."""

    def __init__(
        self,
        enabled: bool = False,
        max_groups: int = 3,
        max_per_group: int = 3,
        strategies: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_groups = max_groups
        self.max_per_group = max_per_group
        self.strategies = strategies or [
            "cross_doc_type",
            "component_link",
            "sequential",
        ]

    @classmethod
    def from_settings(cls, settings: "RAGSettings") -> "RelationExpander":
        """RAGSettings에서 RelationExpander 생성."""
        strategies = [
            s.strip()
            for s in settings.relation_expand_strategies.split(",")
            if s.strip()
        ]
        return cls(
            enabled=settings.relation_expand_enabled,
            max_groups=settings.relation_expand_max_groups,
            max_per_group=settings.relation_expand_max_per_group,
            strategies=strategies,
        )

    def expand(
        self,
        results: list["RetrievalResult"],
        es_client: Any,
        content_index: str,
    ) -> RelationExpandResult:
        """검색 결과에서 관계 확장 수행.

        Args:
            results: 검색+리랭크 후 결과
            es_client: Elasticsearch client
            content_index: content 인덱스명

        Returns:
            RelationExpandResult
        """
        if not self.enabled or not results:
            return RelationExpandResult(
                original_results=results,
                expanded_groups=[],
            )

        groups: list[RelationGroup] = []
        seen_chunk_ids: set[str] = {
            (r.metadata or {}).get("chunk_id", r.doc_id)
            for r in results
        }

        for strategy in self.strategies:
            if len(groups) >= self.max_groups:
                break

            handler = self._STRATEGY_MAP.get(strategy)
            if handler is None:
                logger.warning("Unknown relation strategy: %s", strategy)
                continue

            new_groups = handler(
                self, results, es_client, content_index, seen_chunk_ids
            )
            for g in new_groups:
                if len(groups) >= self.max_groups:
                    break
                groups.append(g)
                for r in g.related_results:
                    cid = (r.metadata or {}).get("chunk_id", r.doc_id)
                    seen_chunk_ids.add(cid)

        if groups:
            total_expanded = sum(len(g.related_results) for g in groups)
            logger.info(
                "[RelationExpander] %d groups, %d expanded chunks (strategies=%s)",
                len(groups),
                total_expanded,
                ",".join(self.strategies),
            )

        return RelationExpandResult(
            original_results=results,
            expanded_groups=groups,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _expand_cross_doc_type(
        self,
        results: list["RetrievalResult"],
        es_client: Any,
        content_index: str,
        seen: set[str],
    ) -> list[RelationGroup]:
        """같은 device_name + 다른 doc_type의 chunk를 가져온다."""
        from backend.llm_infrastructure.retrieval.base import RetrievalResult as RR

        groups: list[RelationGroup] = []
        processed_devices: set[str] = set()

        for r in results:
            meta = r.metadata or {}
            device_name = str(meta.get("device_name", "")).strip()
            doc_type = str(meta.get("doc_type", "")).strip()
            chunk_id = str(meta.get("chunk_id", r.doc_id))

            if not device_name or device_name in processed_devices:
                continue
            processed_devices.add(device_name)

            # Search for same device, different doc_type
            query: dict[str, Any] = {
                "bool": {
                    "filter": [
                        {"term": {"device_name": device_name}},
                    ],
                    "must_not": [
                        {"term": {"doc_type": doc_type}},
                    ],
                }
            }

            try:
                resp = es_client.search(
                    index=content_index,
                    body={
                        "query": query,
                        "size": self.max_per_group,
                        "_source": _RELATION_SOURCE_FIELDS,
                        "track_total_hits": False,
                    },
                )
            except Exception as exc:
                logger.warning("[RelationExpander] cross_doc_type search failed: %s", exc)
                continue

            hits = resp.get("hits", {}).get("hits", [])
            related = _hits_to_results(hits, seen)
            if related:
                groups.append(RelationGroup(
                    trigger_doc_id=r.doc_id,
                    trigger_chunk_id=chunk_id,
                    related_results=related[:self.max_per_group],
                    relation_type="cross_doc_type",
                ))
                logger.info(
                    "[RelationExpander] cross_doc_type: device=%s, found %d related chunks",
                    device_name, len(related),
                )

            if len(groups) >= self.max_groups:
                break

        return groups

    def _expand_equip_history(
        self,
        results: list["RetrievalResult"],
        es_client: Any,
        content_index: str,
        seen: set[str],
    ) -> list[RelationGroup]:
        """같은 equip_id의 과거 MyService 이력을 가져온다."""
        groups: list[RelationGroup] = []
        processed_equips: set[str] = set()

        for r in results:
            meta = r.metadata or {}
            equip_id = str(meta.get("equip_id", "")).strip()
            chunk_id = str(meta.get("chunk_id", r.doc_id))

            if not equip_id or equip_id in processed_equips:
                continue
            processed_equips.add(equip_id)

            query: dict[str, Any] = {
                "bool": {
                    "filter": [
                        {"term": {"equip_id": equip_id}},
                        {"term": {"doc_type": "myservice"}},
                    ],
                }
            }

            try:
                resp = es_client.search(
                    index=content_index,
                    body={
                        "query": query,
                        "size": self.max_per_group,
                        "sort": [{"created_at": {"order": "desc", "unmapped_type": "date"}}],
                        "_source": _RELATION_SOURCE_FIELDS,
                        "track_total_hits": False,
                    },
                )
            except Exception as exc:
                logger.warning("[RelationExpander] equip_history search failed: %s", exc)
                continue

            hits = resp.get("hits", {}).get("hits", [])
            related = _hits_to_results(hits, seen)
            if related:
                groups.append(RelationGroup(
                    trigger_doc_id=r.doc_id,
                    trigger_chunk_id=chunk_id,
                    related_results=related[:self.max_per_group],
                    relation_type="equip_history",
                ))
                logger.info(
                    "[RelationExpander] equip_history: equip_id=%s, found %d related chunks",
                    equip_id, len(related),
                )

            if len(groups) >= self.max_groups:
                break

        return groups

    def _expand_component_link(
        self,
        results: list["RetrievalResult"],
        es_client: Any,
        content_index: str,
        seen: set[str],
    ) -> list[RelationGroup]:
        """같은 components 필드를 가진 다른 문서의 chunk를 가져온다."""
        groups: list[RelationGroup] = []

        # Collect all components from top results
        all_components: set[str] = set()
        trigger_doc_ids: set[str] = set()
        trigger_chunk_id = ""

        for r in results:
            meta = r.metadata or {}
            components = meta.get("components", [])
            if isinstance(components, str):
                components = [components]
            if components:
                all_components.update(components)
                trigger_doc_ids.add(r.doc_id)
                if not trigger_chunk_id:
                    trigger_chunk_id = str(meta.get("chunk_id", r.doc_id))

        if not all_components:
            return groups

        # Search for chunks with same components but different doc_id
        must_not = [{"term": {"doc_id": did}} for did in trigger_doc_ids]
        query: dict[str, Any] = {
            "bool": {
                "filter": [
                    {"terms": {"components": list(all_components)}},
                ],
                "must_not": must_not,
            },
        }

        try:
            resp = es_client.search(
                index=content_index,
                body={
                    "query": query,
                    "size": self.max_per_group,
                    "_source": _RELATION_SOURCE_FIELDS,
                    "track_total_hits": False,
                },
            )
        except Exception as exc:
            logger.warning("[RelationExpander] component_link search failed: %s", exc)
            return groups

        hits = resp.get("hits", {}).get("hits", [])
        related = _hits_to_results(hits, seen)
        if related:
            groups.append(RelationGroup(
                trigger_doc_id=list(trigger_doc_ids)[0] if trigger_doc_ids else "",
                trigger_chunk_id=trigger_chunk_id,
                related_results=related[:self.max_per_group],
                relation_type="component_link",
            ))
            logger.info(
                "[RelationExpander] component_link: components=%s, found %d related chunks",
                list(all_components)[:3], len(related),
            )

        return groups

    def _expand_sequential(
        self,
        results: list["RetrievalResult"],
        es_client: Any,
        content_index: str,
        seen: set[str],
    ) -> list[RelationGroup]:
        """같은 doc_id 내 앞뒤 페이지 chunk를 가져온다."""
        groups: list[RelationGroup] = []
        processed_docs: set[str] = set()

        for r in results:
            meta = r.metadata or {}
            doc_id = r.doc_id
            page = meta.get("page")
            chunk_id = str(meta.get("chunk_id", doc_id))

            if not doc_id or doc_id in processed_docs or page is None:
                continue
            processed_docs.add(doc_id)

            try:
                page_int = int(page)
            except (TypeError, ValueError):
                continue

            # Fetch adjacent pages (page-1, page+1, page+2)
            adjacent_pages = [p for p in [page_int - 1, page_int + 1, page_int + 2] if p > 0]
            if not adjacent_pages:
                continue

            query: dict[str, Any] = {
                "bool": {
                    "filter": [
                        {"term": {"doc_id": doc_id}},
                        {"terms": {"page": adjacent_pages}},
                    ],
                }
            }

            try:
                resp = es_client.search(
                    index=content_index,
                    body={
                        "query": query,
                        "size": self.max_per_group,
                        "sort": [{"page": "asc"}],
                        "_source": _RELATION_SOURCE_FIELDS,
                        "track_total_hits": False,
                    },
                )
            except Exception as exc:
                logger.warning("[RelationExpander] sequential search failed: %s", exc)
                continue

            hits = resp.get("hits", {}).get("hits", [])
            related = _hits_to_results(hits, seen)
            if related:
                groups.append(RelationGroup(
                    trigger_doc_id=doc_id,
                    trigger_chunk_id=chunk_id,
                    related_results=related[:self.max_per_group],
                    relation_type="sequential",
                ))

            if len(groups) >= self.max_groups:
                break

        return groups

    # Strategy dispatch map
    _STRATEGY_MAP = {
        "cross_doc_type": _expand_cross_doc_type,
        "equip_history": _expand_equip_history,
        "component_link": _expand_component_link,
        "sequential": _expand_sequential,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_RELATION_SOURCE_FIELDS = [
    "doc_id",
    "chunk_id",
    "content",
    "search_text",
    "page",
    "doc_type",
    "device_name",
    "equip_id",
    "chapter",
    "chunk_summary",
    "components",
]


def _hits_to_results(
    hits: list[dict[str, Any]],
    seen: set[str],
) -> list["RetrievalResult"]:
    """Convert ES hits to RetrievalResult, excluding already-seen chunk_ids."""
    from backend.llm_infrastructure.retrieval.base import RetrievalResult

    results: list[RetrievalResult] = []
    for hit in hits:
        source = hit.get("_source", {})
        chunk_id = str(source.get("chunk_id") or hit.get("_id") or "")
        if chunk_id in seen:
            continue

        doc_id = str(source.get("doc_id") or chunk_id)
        content = str(source.get("search_text") or source.get("content") or "")
        raw_text = str(source.get("content") or content)

        metadata = {
            k: v for k, v in source.items()
            if k not in {"content", "search_text"}
        }
        metadata["chunk_id"] = chunk_id

        results.append(RetrievalResult(
            doc_id=doc_id,
            content=content,
            score=0.0,  # Relation-expanded results don't have retrieval scores
            metadata=metadata,
            raw_text=raw_text,
        ))

    return results


__all__ = [
    "RelationExpander",
    "RelationExpandResult",
    "RelationGroup",
]
