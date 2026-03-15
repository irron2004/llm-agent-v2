from pathlib import Path
import sys
from typing import Any, cast

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from llm_infrastructure.llm.langgraph_agent import AgentState, auto_parse_node

_DUMMY = cast(Any, None)


def test_auto_parse_node_emits_event_and_message_without_device_doc_type_match() -> None:
    state: AgentState = {"query": "How can I reduce process drift?"}

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["SUPRA N"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["detected_language"] == "en"
    assert result["auto_parsed_devices"] == []
    assert result["auto_parsed_doc_types"] == []
    assert isinstance(result.get("auto_parse_message"), str)
    assert "lang:" in result["auto_parse_message"]
    assert result.get("_events")
    assert result["_events"][0]["type"] == "auto_parse"
    assert result["_events"][0]["language"] == "en"


def test_auto_parse_node_equip_id_parsing_is_disabled() -> None:
    state: AgentState = {"query": "equip_id: epag50 관련 gcb 문서 찾아줘"}

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["SUPRA N"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["auto_parsed_equip_id"] is None
    assert result["auto_parsed_equip_ids"] == []
    assert result["selected_equip_ids"] == []
    assert result["_events"][0]["equip_id"] is None


def test_auto_parse_node_ignores_short_component_like_device_labels() -> None:
    state: AgentState = {"query": "how to replacement apc position"}

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["APC", "SUPRA N", "ALL", "etc"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["auto_parsed_device"] is None
    assert result["auto_parsed_devices"] == []


def test_auto_parse_node_sticky_keeps_previous_selections_when_no_detection() -> None:
    state: AgentState = {
        "query": "How can I reduce process drift?",
        "selected_devices": ["SUPRA N"],
        "selected_doc_types": ["gcb"],
        "selected_equip_ids": ["EPAG49"],
    }

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["SUPRA N", "ALL"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["selected_devices"] == ["SUPRA N"]
    assert result["selected_doc_types"] == ["gcb"]
    assert result["selected_equip_ids"] == []
    assert result["parsed_query"]["device_names"] == ["SUPRA N"]
    assert result["parsed_query"]["doc_types"] == ["gcb"]
    assert result["parsed_query"]["equip_ids"] == []
    assert result["parsed_query"]["selected_devices"] == ["SUPRA N"]
    assert result["parsed_query"]["selected_doc_types"] == ["gcb"]
    assert result["parsed_query"]["selected_equip_ids"] == []
    assert result["device_selection_skipped"] is False
    assert result["doc_type_selection_skipped"] is False
    assert result["_events"][0]["type"] == "auto_parse"
    assert result["_events"][0]["devices"] == ["SUPRA N"]
    assert result["_events"][0]["doc_types"] == ["gcb"]
    assert result["_events"][0]["equip_ids"] == ["EPAG49"]
    assert "SUPRA N" in result["_events"][0]["message"]
    assert "gcb" in result["_events"][0]["message"]
    assert "EPAG49" in result["_events"][0]["message"]


def test_auto_parse_node_replaces_selections_when_new_detection() -> None:
    state: AgentState = {
        "query": "SUPRA N gcb equip_id: epag50 문서 찾아줘",
        "selected_devices": ["OLD_DEVICE"],
        "selected_doc_types": ["sop"],
        "selected_equip_ids": ["EPAG49"],
    }

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["SUPRA N", "OLD_DEVICE"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["selected_devices"] == ["SUPRA N"]
    assert result["selected_doc_types"] == ["gcb"]
    assert result["selected_equip_ids"] == []
    assert result["parsed_query"]["device_names"] == ["SUPRA N"]
    assert result["parsed_query"]["doc_types"] == ["gcb"]
    assert result["parsed_query"]["equip_ids"] == []
    assert result["parsed_query"]["selected_devices"] == ["SUPRA N"]
    assert result["parsed_query"]["selected_doc_types"] == ["gcb"]
    assert result["parsed_query"]["selected_equip_ids"] == []
    assert result["device_selection_skipped"] is False
    assert result["doc_type_selection_skipped"] is False
    assert result["_events"][0]["type"] == "auto_parse"
    assert result["_events"][0]["device"] == "SUPRA N"
    assert result["_events"][0]["doc_type"] == "gcb"
    assert result["_events"][0]["equip_id"] == "EPAG49"


def test_auto_parse_node_strict_scope_can_narrow_when_new_doc_type_detected() -> None:
    state: AgentState = {
        "query": "gcb 문서만 보여줘",
        "selected_doc_types": ["myservice", "gcb", "ts"],
        "selected_doc_types_strict": True,
    }

    result = auto_parse_node(
        state,
        llm=_DUMMY,
        spec=_DUMMY,
        device_names=["SUPRA N"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["auto_parsed_doc_types"] == ["gcb"]
    assert result["selected_doc_types"] == ["gcb"]
    assert result["parsed_query"]["doc_types"] == ["gcb"]
    assert result["parsed_query"]["selected_doc_types"] == ["gcb"]
