"""Relation expander unit tests.

Tests:
- RelationExpander: disabled, enabled, from_settings, strategy dispatch
- RelationExpandResult: dedup, all_results ordering, metadata tagging
- Individual strategies: cross_doc_type, equip_history, component_link, sequential
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.postprocessors.relation_expander import (
    RelationExpander,
    RelationExpandResult,
    RelationGroup,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_result(
    doc_id: str = "doc1",
    chunk_id: str = "chunk1",
    content: str = "test content",
    score: float = 1.0,
    doc_type: str = "sop",
    device_name: str = "DEVICE_A",
    page: int = 1,
    components: list[str] | None = None,
    equip_id: str = "",
) -> RetrievalResult:
    metadata = {
        "chunk_id": chunk_id,
        "doc_type": doc_type,
        "device_name": device_name,
        "page": page,
    }
    if components:
        metadata["components"] = components
    if equip_id:
        metadata["equip_id"] = equip_id
    return RetrievalResult(
        doc_id=doc_id,
        content=content,
        score=score,
        metadata=metadata,
    )


def _mock_es_search(hits_map: dict[str, list[dict]] | None = None):
    """Create a mock ES client.

    hits_map: maps a strategy hint to ES hits. If None, returns empty.
    """
    es = MagicMock()
    all_hits = []
    if hits_map:
        for hits in hits_map.values():
            all_hits.extend(hits)

    def search_side_effect(**kwargs):
        return {"hits": {"hits": all_hits}}

    es.search.side_effect = search_side_effect
    return es


def _es_hit(
    chunk_id: str,
    doc_id: str = "related_doc",
    content: str = "related content",
    doc_type: str = "ts",
    device_name: str = "DEVICE_A",
    page: int = 1,
    components: list[str] | None = None,
) -> dict:
    """Build a fake ES hit dict."""
    source = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content": content,
        "doc_type": doc_type,
        "device_name": device_name,
        "page": page,
    }
    if components:
        source["components"] = components
    return {"_id": chunk_id, "_source": source}


# =========================================================================
# RelationExpander Core Tests
# =========================================================================

class TestRelationExpanderCore:

    def test_disabled_returns_original(self):
        results = [_make_result()]
        expander = RelationExpander(enabled=False)
        expand_result = expander.expand(results, MagicMock(), "idx")
        assert expand_result.original_results == results
        assert expand_result.expanded_groups == []

    def test_empty_results(self):
        expander = RelationExpander(enabled=True)
        expand_result = expander.expand([], MagicMock(), "idx")
        assert expand_result.original_results == []
        assert expand_result.expanded_groups == []

    def test_from_settings(self):
        settings = MagicMock()
        settings.relation_expand_enabled = True
        settings.relation_expand_max_groups = 5
        settings.relation_expand_max_per_group = 4
        settings.relation_expand_strategies = "cross_doc_type,component_link"

        expander = RelationExpander.from_settings(settings)
        assert expander.enabled is True
        assert expander.max_groups == 5
        assert expander.max_per_group == 4
        assert expander.strategies == ["cross_doc_type", "component_link"]

    def test_from_settings_empty_strategies_falls_back_to_default(self):
        """Empty strategies string falls back to default strategies."""
        settings = MagicMock()
        settings.relation_expand_enabled = False
        settings.relation_expand_max_groups = 3
        settings.relation_expand_max_per_group = 3
        settings.relation_expand_strategies = ""

        expander = RelationExpander.from_settings(settings)
        # Empty list is falsy -> __init__ applies default strategies
        assert expander.strategies == ["cross_doc_type", "component_link", "sequential"]

    def test_unknown_strategy_skipped(self):
        """Unknown strategy should log warning and skip."""
        results = [_make_result()]
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": []}}

        expander = RelationExpander(
            enabled=True, strategies=["nonexistent_strategy"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert expand_result.expanded_groups == []

    def test_max_groups_limit(self):
        """Should not exceed max_groups."""
        results = [
            _make_result(doc_id="d1", chunk_id="c1", device_name="DEV_A"),
            _make_result(doc_id="d2", chunk_id="c2", device_name="DEV_B"),
            _make_result(doc_id="d3", chunk_id="c3", device_name="DEV_C"),
        ]
        es = MagicMock()
        # Return different related hits for each device
        call_count = [0]

        def search_fn(**kwargs):
            call_count[0] += 1
            return {"hits": {"hits": [
                _es_hit(f"rel_{call_count[0]}", doc_id=f"rdoc{call_count[0]}")
            ]}}

        es.search.side_effect = search_fn

        expander = RelationExpander(
            enabled=True,
            max_groups=2,
            strategies=["cross_doc_type"],
        )
        expand_result = expander.expand(results, es, "idx")
        assert len(expand_result.expanded_groups) <= 2


# =========================================================================
# RelationExpandResult Tests
# =========================================================================

class TestRelationExpandResult:

    def test_all_results_dedup(self):
        """Duplicate chunk_ids should be removed."""
        original = [_make_result(chunk_id="c1"), _make_result(chunk_id="c2")]
        related = [
            RetrievalResult(
                doc_id="doc3", content="text", score=0.0,
                metadata={"chunk_id": "c1"},  # duplicate
            ),
            RetrievalResult(
                doc_id="doc4", content="text", score=0.0,
                metadata={"chunk_id": "c3"},
            ),
        ]
        group = RelationGroup(
            trigger_doc_id="doc1",
            trigger_chunk_id="c1",
            related_results=related,
            relation_type="component_link",
        )
        result = RelationExpandResult(
            original_results=original,
            expanded_groups=[group],
        )
        all_r = result.all_results()
        chunk_ids = [(r.metadata or {}).get("chunk_id") for r in all_r]
        assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates

    def test_all_results_original_first(self):
        """Original results should come before expanded."""
        original = [_make_result(chunk_id="c1", score=5.0)]
        related = [
            RetrievalResult(
                doc_id="doc2", content="expanded", score=0.0,
                metadata={"chunk_id": "c_exp", "doc_type": "sop"},
            ),
        ]
        group = RelationGroup(
            trigger_doc_id="doc1",
            trigger_chunk_id="c1",
            related_results=related,
            relation_type="sequential",
        )
        result = RelationExpandResult(
            original_results=original,
            expanded_groups=[group],
        )
        all_r = result.all_results()
        assert len(all_r) == 2
        assert (all_r[0].metadata or {}).get("chunk_id") == "c1"
        assert (all_r[1].metadata or {}).get("chunk_id") == "c_exp"

    def test_cross_type_excluded_from_all_results(self):
        """Different doc_type expanded results should not appear in all_results."""
        original = [_make_result(chunk_id="c1", doc_type="sop")]
        related = [
            RetrievalResult(
                doc_id="doc2", content="text", score=0.0,
                metadata={"chunk_id": "c_cross", "doc_type": "ts"},
            ),
        ]
        group = RelationGroup(
            trigger_doc_id="doc1",
            trigger_chunk_id="c1",
            related_results=related,
            relation_type="cross_doc_type",
        )
        result = RelationExpandResult(
            original_results=original,
            expanded_groups=[group],
        )
        all_r = result.all_results()
        assert len(all_r) == 1  # only original, cross-type excluded
        suggestions = result.cross_type_suggestions()
        assert "ts" in suggestions
        assert len(suggestions["ts"]) == 1

    def test_expanded_tagged_with_relation_type(self):
        """Expanded results should have relation_type and relation_trigger."""
        original = [_make_result(chunk_id="c1")]
        related = [
            RetrievalResult(
                doc_id="doc2", content="text", score=0.0,
                metadata={"chunk_id": "c_exp", "doc_type": "sop"},
            ),
        ]
        group = RelationGroup(
            trigger_doc_id="doc1",
            trigger_chunk_id="c1",
            related_results=related,
            relation_type="component_link",
        )
        result = RelationExpandResult(
            original_results=original,
            expanded_groups=[group],
        )
        all_r = result.all_results()
        expanded = all_r[1]
        assert expanded.metadata["relation_type"] == "component_link"
        assert expanded.metadata["relation_trigger"] == "c1"


# =========================================================================
# Strategy-specific Tests
# =========================================================================

class TestCrossDocTypeStrategy:

    def test_finds_different_doc_type(self):
        """Should search for same device, different doc_type."""
        results = [
            _make_result(
                doc_id="sop_doc", chunk_id="c1",
                device_name="ETCH_01", doc_type="sop",
            ),
        ]
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": [
            _es_hit("ts_chunk", doc_id="ts_doc", doc_type="ts", device_name="ETCH_01"),
        ]}}

        expander = RelationExpander(
            enabled=True, strategies=["cross_doc_type"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert len(expand_result.expanded_groups) == 1
        assert expand_result.expanded_groups[0].relation_type == "cross_doc_type"

    def test_skips_empty_device_name(self):
        results = [_make_result(device_name="")]
        es = MagicMock()

        expander = RelationExpander(
            enabled=True, strategies=["cross_doc_type"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert expand_result.expanded_groups == []
        es.search.assert_not_called()


class TestComponentLinkStrategy:

    def test_finds_shared_components(self):
        results = [
            _make_result(
                chunk_id="c1", doc_id="doc1",
                components=["ROBOT", "CHUCK"],
            ),
        ]
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": [
            _es_hit("c_rel", doc_id="doc2", components=["ROBOT"]),
        ]}}

        expander = RelationExpander(
            enabled=True, strategies=["component_link"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert len(expand_result.expanded_groups) == 1
        assert expand_result.expanded_groups[0].relation_type == "component_link"

    def test_no_components_skips(self):
        results = [_make_result(components=None)]
        es = MagicMock()

        expander = RelationExpander(
            enabled=True, strategies=["component_link"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert expand_result.expanded_groups == []
        es.search.assert_not_called()


class TestSequentialStrategy:

    def test_finds_adjacent_pages(self):
        results = [
            _make_result(doc_id="doc1", chunk_id="c1", page=5),
        ]
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": [
            _es_hit("c_p4", doc_id="doc1", page=4),
            _es_hit("c_p6", doc_id="doc1", page=6),
        ]}}

        expander = RelationExpander(
            enabled=True, strategies=["sequential"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert len(expand_result.expanded_groups) == 1
        assert expand_result.expanded_groups[0].relation_type == "sequential"
        assert len(expand_result.expanded_groups[0].related_results) == 2

    def test_page_none_skips(self):
        results = [_make_result(page=None)]
        # Override metadata to have page=None
        results[0].metadata["page"] = None
        es = MagicMock()

        expander = RelationExpander(
            enabled=True, strategies=["sequential"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert expand_result.expanded_groups == []


class TestEquipHistoryStrategy:

    def test_finds_equip_history(self):
        results = [
            _make_result(
                doc_id="doc1", chunk_id="c1",
                equip_id="EQ_001",
            ),
        ]
        es = MagicMock()
        es.search.return_value = {"hits": {"hits": [
            _es_hit("ms_chunk", doc_id="ms_doc", doc_type="myservice"),
        ]}}

        expander = RelationExpander(
            enabled=True, strategies=["equip_history"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert len(expand_result.expanded_groups) == 1
        assert expand_result.expanded_groups[0].relation_type == "equip_history"

    def test_no_equip_id_skips(self):
        results = [_make_result(equip_id="")]
        es = MagicMock()

        expander = RelationExpander(
            enabled=True, strategies=["equip_history"]
        )
        expand_result = expander.expand(results, es, "idx")
        assert expand_result.expanded_groups == []
        es.search.assert_not_called()
