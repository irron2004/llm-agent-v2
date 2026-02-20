from backend.llm_infrastructure.llm.langgraph_agent import auto_parse_node


def test_auto_parse_node_emits_event_and_message_without_device_doc_type_match() -> None:
    state = {"query": "How can I reduce process drift?"}

    result = auto_parse_node(
        state,
        llm=None,  # type: ignore[arg-type]
        spec=None,  # type: ignore[arg-type]
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


def test_auto_parse_node_extracts_equip_id_from_query() -> None:
    state = {"query": "equip_id: epag50 관련 gcb 문서 찾아줘"}

    result = auto_parse_node(
        state,
        llm=None,  # type: ignore[arg-type]
        spec=None,  # type: ignore[arg-type]
        device_names=["SUPRA N"],
        doc_type_names=["myservice", "ts", "gcb", "sop", "setup"],
    )

    assert result["auto_parsed_equip_id"] == "EPAG50"
    assert result["auto_parsed_equip_ids"] == ["EPAG50"]
    assert result["selected_equip_ids"] == ["EPAG50"]
    assert "EPAG50" in result["auto_parse_message"]
    assert result["_events"][0]["equip_id"] == "EPAG50"
