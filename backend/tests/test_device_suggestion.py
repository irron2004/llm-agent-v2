"""Tests for device suggestion feature."""
import pytest
from unittest.mock import MagicMock

from backend.llm_infrastructure.llm.langgraph_agent import (
    _collect_suggested_devices,
    auto_parse_node,
)


class TestCollectSuggestedDevices:
    """Test _collect_suggested_devices function."""

    def test_basic_collection(self):
        """Should collect and count device_names."""
        docs = [
            MagicMock(metadata={"device_name": "SUPRA XP"}),
            MagicMock(metadata={"device_name": "SUPRA XP"}),
            MagicMock(metadata={"device_name": "SUPRA N"}),
            MagicMock(metadata={"device_name": "PRISM"}),
        ]

        result = _collect_suggested_devices(docs)

        assert len(result) == 3
        assert result[0] == {"name": "SUPRA XP", "count": 2}
        assert result[1] == {"name": "SUPRA N", "count": 1}
        assert result[2] == {"name": "PRISM", "count": 1}

    def test_excludes_empty_names(self):
        """Should exclude empty device_names."""
        docs = [
            MagicMock(metadata={"device_name": "SUPRA XP"}),
            MagicMock(metadata={"device_name": ""}),
            MagicMock(metadata={"device_name": None}),
            MagicMock(metadata={}),
        ]

        result = _collect_suggested_devices(docs)

        assert len(result) == 1
        assert result[0]["name"] == "SUPRA XP"

    def test_excludes_dummy_names(self):
        """Should exclude ALL, etc, and other dummy names."""
        docs = [
            MagicMock(metadata={"device_name": "SUPRA XP"}),
            MagicMock(metadata={"device_name": "ALL"}),
            MagicMock(metadata={"device_name": "etc"}),
            MagicMock(metadata={"device_name": "ETC"}),
            MagicMock(metadata={"device_name": "N/A"}),
        ]

        result = _collect_suggested_devices(docs)

        assert len(result) == 1
        assert result[0]["name"] == "SUPRA XP"

    def test_empty_docs(self):
        """Should return empty list for empty docs."""
        result = _collect_suggested_devices([])
        assert result == []

    def test_dict_docs(self):
        """Should support dict format docs."""
        docs = [
            {"metadata": {"device_name": "SUPRA XP"}},
            {"metadata": {"device_name": "SUPRA N"}},
        ]

        result = _collect_suggested_devices(docs)

        assert len(result) == 2

    def test_sorted_by_count_descending(self):
        """Should sort by count in descending order."""
        docs = [
            MagicMock(metadata={"device_name": "A"}),
            MagicMock(metadata={"device_name": "B"}),
            MagicMock(metadata={"device_name": "B"}),
            MagicMock(metadata={"device_name": "C"}),
            MagicMock(metadata={"device_name": "C"}),
            MagicMock(metadata={"device_name": "C"}),
        ]

        result = _collect_suggested_devices(docs)

        assert result[0]["name"] == "C"
        assert result[0]["count"] == 3
        assert result[1]["name"] == "B"
        assert result[1]["count"] == 2
        assert result[2]["name"] == "A"
        assert result[2]["count"] == 1


class TestAutoParseNodeSkip:
    """Test auto_parse_node skip_auto_parse behavior."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.fixture
    def mock_spec(self):
        """Create a mock PromptSpec."""
        return MagicMock()

    def test_skip_auto_parse_with_selected_devices(self, mock_llm, mock_spec):
        """Should skip parsing when skip_auto_parse=True."""
        state = {
            "query": "알람 해결 방법",
            "skip_auto_parse": True,
            "selected_devices": ["SUPRA XP"],
        }

        result = auto_parse_node(
            state,
            llm=mock_llm,
            spec=mock_spec,
            device_names=["SUPRA XP", "SUPRA N"],
            doc_type_names=["manual", "ts"],
        )

        assert result["auto_parsed_device"] == "SUPRA XP"
        assert result["auto_parsed_devices"] == ["SUPRA XP"]
        assert "필터 적용" in result["auto_parse_message"]
        # LLM should not be called
        mock_llm.generate.assert_not_called()

    def test_skip_auto_parse_without_devices(self, mock_llm, mock_spec):
        """Should handle skip_auto_parse with empty devices."""
        state = {
            "query": "알람 해결 방법",
            "skip_auto_parse": True,
            "selected_devices": [],
        }

        result = auto_parse_node(
            state,
            llm=mock_llm,
            spec=mock_spec,
            device_names=["SUPRA XP"],
            doc_type_names=[],
        )

        assert result["auto_parsed_device"] is None
        assert result["auto_parsed_devices"] is None
        # LLM should not be called
        mock_llm.generate.assert_not_called()
