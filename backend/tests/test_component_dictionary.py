"""Component dictionary unit tests.

Tests:
- _build_pattern regex generation
- match_components (exact matching, overlap handling, max_components)
- match_components_fuzzy (fuzzy matching, threshold, fallback)
- build_component_dict_from_manifest (blocklist, sorting, min length)
"""

from __future__ import annotations

import pytest

from backend.domain.component_dictionary import (
    ComponentEntry,
    _build_pattern,
    match_components,
    match_components_fuzzy,
)


# =========================================================================
# Helpers
# =========================================================================

def _entry(name: str, module: str = "") -> ComponentEntry:
    return ComponentEntry(canonical=name, pattern=_build_pattern(name), module=module)


def _dict(*names: str) -> list[ComponentEntry]:
    """Build a test dictionary sorted longest-first."""
    entries = [_entry(n) for n in names]
    entries.sort(key=lambda e: len(e.canonical), reverse=True)
    return entries


# =========================================================================
# _build_pattern Tests
# =========================================================================

class TestBuildPattern:

    def test_single_word(self):
        p = _build_pattern("ROBOT")
        assert p.search("ROBOT arm") is not None
        assert p.search("robot arm") is not None  # case insensitive
        assert p.search("no match") is None

    def test_multi_word_space(self):
        p = _build_pattern("BARATRON GAUGE")
        assert p.search("Baratron Gauge 교체") is not None
        assert p.search("BARATRON_GAUGE test") is not None  # underscore also matches
        assert p.search("BARATRON test GAUGE") is None  # non-adjacent

    def test_multi_word_underscore(self):
        p = _build_pattern("HEATER_CHUCK")
        assert p.search("heater chuck") is not None
        assert p.search("HEATER_CHUCK") is not None

    def test_special_chars_escaped(self):
        p = _build_pattern("D-NET")
        # The hyphen in D-NET should be escaped
        assert p.search("D-NET controller") is not None

    def test_empty_returns_pattern(self):
        p = _build_pattern("")
        assert p is not None


# =========================================================================
# match_components Tests
# =========================================================================

class TestMatchComponents:

    def test_single_match(self):
        d = _dict("ROBOT", "CHUCK")
        result = match_components("ROBOT arm 교체 작업", d)
        assert result == ["ROBOT"]

    def test_multiple_matches(self):
        d = _dict("ROBOT", "CHUCK", "MFC")
        result = match_components("ROBOT arm 및 MFC 교체", d)
        assert "ROBOT" in result
        assert "MFC" in result
        assert len(result) == 2

    def test_no_match(self):
        d = _dict("ROBOT", "CHUCK")
        result = match_components("일반적인 텍스트입니다", d)
        assert result == []

    def test_empty_text(self):
        d = _dict("ROBOT")
        assert match_components("", d) == []

    def test_longest_first_priority(self):
        """HEATER CHUCK이 CHUCK보다 먼저 매칭되어야 한다."""
        d = _dict("HEATER CHUCK", "CHUCK")
        result = match_components("HEATER CHUCK 온도 설정", d)
        assert result == ["HEATER CHUCK"]

    def test_overlap_prevention(self):
        """겹치는 위치의 짧은 매칭은 건너뛴다."""
        d = _dict("APC VALVE", "APC", "VALVE")
        result = match_components("APC VALVE 교체", d)
        assert result == ["APC VALVE"]

    def test_non_overlapping_allowed(self):
        """겹치지 않는 매칭은 둘 다 포함."""
        d = _dict("ROBOT", "MFC")
        result = match_components("ROBOT 정비 후 MFC 교체", d)
        assert len(result) == 2

    def test_max_components(self):
        d = _dict("A1", "B1", "C1", "D1", "E1")
        text = "A1 B1 C1 D1 E1 all present"
        result = match_components(text, d, max_components=2)
        assert len(result) == 2

    def test_case_insensitive(self):
        d = _dict("BARATRON GAUGE")
        result = match_components("baratron gauge 점검", d)
        assert result == ["BARATRON GAUGE"]


# =========================================================================
# match_components_fuzzy Tests
# =========================================================================

class TestMatchComponentsFuzzy:

    def test_exact_match_via_fuzzy(self):
        d = _dict("BARATRON GAUGE", "ROBOT")
        result = match_components_fuzzy("BARATRON GAUGE 교체", d, threshold=80)
        assert "BARATRON GAUGE" in result

    def test_threshold_filtering(self):
        d = _dict("BARATRON GAUGE")
        # Very strict threshold should filter most partial matches
        low = match_components_fuzzy("바라트론", d, threshold=50)
        high = match_components_fuzzy("바라트론", d, threshold=99)
        assert len(low) >= len(high)

    def test_empty_text(self):
        d = _dict("ROBOT")
        assert match_components_fuzzy("", d) == []

    def test_max_components(self):
        d = _dict("ROBOT", "CHUCK", "MFC", "PIN", "EPD")
        result = match_components_fuzzy(
            "ROBOT CHUCK MFC PIN EPD", d, threshold=80, max_components=2
        )
        assert len(result) <= 2


# =========================================================================
# Integration-style Tests
# =========================================================================

class TestComponentDictIntegration:

    def test_real_dict_loads(self):
        """실제 component_dictionary.json 로드 테스트."""
        from backend.domain.component_dictionary import get_component_dict

        comp_dict = get_component_dict()
        assert len(comp_dict) > 100  # Should have 383 entries

    def test_real_dict_match(self):
        """실제 사전으로 매칭 테스트."""
        from backend.domain.component_dictionary import get_component_dict

        comp_dict = get_component_dict()
        result = match_components("BARATRON GAUGE 교체 작업", comp_dict)
        assert "BARATRON GAUGE" in result

    def test_blocklist_excluded(self):
        """ALL, ETC, PM 등 blocklist 항목은 사전에 없어야 한다."""
        from backend.domain.component_dictionary import get_component_dict

        comp_dict = get_component_dict()
        names = {e.canonical for e in comp_dict}
        assert "ALL" not in names
        assert "ETC" not in names
        assert "PM" not in names
