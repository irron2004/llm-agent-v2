"""Tests for equip_id lookup-based extraction from queries."""

from backend.llm_infrastructure.llm.langgraph_agent import (
    _extract_equip_ids_by_lookup,
    _extract_equip_ids_from_query,
)

# Simulated known equip_id set (representative subset)
KNOWN_EQUIP_IDS = {
    "EPAG50", "EPAG51", "5EBP0701", "DES02", "DES03",
    "HDP01", "CVD100", "3PVD0501", "ABC1234",
}


class TestExtractEquipIdsByLookup:
    """Tests for _extract_equip_ids_by_lookup (direct lookup)."""

    def test_single_token_match(self):
        assert _extract_equip_ids_by_lookup("EPAG50 장비 캘리브레이션 방법", KNOWN_EQUIP_IDS) == ["EPAG50"]

    def test_single_token_case_insensitive(self):
        assert _extract_equip_ids_by_lookup("epag50 에러 조치", KNOWN_EQUIP_IDS) == ["EPAG50"]

    def test_multiple_tokens_match(self):
        result = _extract_equip_ids_by_lookup("EPAG50 과 EPAG51 비교", KNOWN_EQUIP_IDS)
        assert result == ["EPAG50", "EPAG51"]

    def test_max_two_results(self):
        result = _extract_equip_ids_by_lookup("EPAG50 EPAG51 DES02 장비", KNOWN_EQUIP_IDS)
        assert len(result) <= 2

    def test_adjacent_token_combination(self):
        """'DES 02' should combine to 'DES02' and match."""
        result = _extract_equip_ids_by_lookup("DES 02 장비 상태", KNOWN_EQUIP_IDS)
        assert "DES02" in result

    def test_three_token_combination(self):
        """'5 EBP 0701' should combine to '5EBP0701' and match."""
        result = _extract_equip_ids_by_lookup("5 EBP 0701 에러", KNOWN_EQUIP_IDS)
        assert "5EBP0701" in result

    def test_no_match_returns_empty(self):
        assert _extract_equip_ids_by_lookup("장비 상태 확인", KNOWN_EQUIP_IDS) == []

    def test_device_name_not_matched(self):
        """Device names like 'SUPRA N' should not match equip_ids."""
        assert _extract_equip_ids_by_lookup("SUPRA N 캘리브레이션", KNOWN_EQUIP_IDS) == []

    def test_empty_query(self):
        assert _extract_equip_ids_by_lookup("", KNOWN_EQUIP_IDS) == []

    def test_empty_known_set(self):
        assert _extract_equip_ids_by_lookup("EPAG50 에러", set()) == []

    def test_punctuation_stripped(self):
        assert _extract_equip_ids_by_lookup("EPAG50, 에러 코드", KNOWN_EQUIP_IDS) == ["EPAG50"]


class TestExtractEquipIdsFromQuery:
    """Tests for _extract_equip_ids_from_query (with and without lookup)."""

    def test_with_known_equip_ids_uses_lookup(self):
        result = _extract_equip_ids_from_query("EPAG50 장비 캘리브레이션", known_equip_ids=KNOWN_EQUIP_IDS)
        assert result == ["EPAG50"]

    def test_with_known_equip_ids_adjacent_tokens(self):
        result = _extract_equip_ids_from_query("DES 02 에러", known_equip_ids=KNOWN_EQUIP_IDS)
        assert "DES02" in result

    def test_without_known_equip_ids_falls_back_to_regex(self):
        """Without known set, should fall back to regex patterns."""
        result = _extract_equip_ids_from_query("장비번호 EPAG50 에러 조치", known_equip_ids=None)
        assert result == ["EPAG50"]

    def test_no_equip_id_in_query(self):
        result = _extract_equip_ids_from_query("장비 상태 확인", known_equip_ids=KNOWN_EQUIP_IDS)
        assert result == []

    def test_lookup_miss_falls_back_to_regex(self):
        """If token not in known set, regex fallback should still work for explicit patterns."""
        result = _extract_equip_ids_from_query(
            "장비번호 UNKNOWN123",
            known_equip_ids=KNOWN_EQUIP_IDS,
        )
        # Regex fallback should catch "UNKNOWN123" via explicit cue pattern
        assert result == ["UNKNOWN123"]

    def test_5ebp0701_direct(self):
        result = _extract_equip_ids_from_query("5EBP0701 에러 조치", known_equip_ids=KNOWN_EQUIP_IDS)
        assert result == ["5EBP0701"]
