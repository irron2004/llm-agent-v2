from backend.llm_infrastructure.llm.langgraph_agent import _is_garbage_query, _parse_queries


def test_parse_queries_filters_query_number_placeholders() -> None:
    raw = """
    {
      "queries": ["query1", "query2", "how to increase ashing rate"]
    }
    """

    parsed = _parse_queries(raw)

    assert parsed == ["how to increase ashing rate"]


def test_parse_queries_filters_plain_placeholder_lines() -> None:
    raw = "query1\nquery2\nquery3"

    parsed = _parse_queries(raw)

    assert parsed == []


def test_parse_queries_strips_regeneration_prefixes() -> None:
    raw = "[Regenerate with All equipment / sop] [Regenerate with All equipment / myservice] how can I increase the ashing rate if it is currently too low?"

    parsed = _parse_queries(raw)

    assert parsed == ["how can I increase the ashing rate if it is currently too low?"]


def test_parse_queries_filters_ellipsis_placeholder() -> None:
    raw = "...\n…\nashing rate too low troubleshooting"

    parsed = _parse_queries(raw)

    assert parsed == ["ashing rate too low troubleshooting"]


def test_is_garbage_query_detects_placeholder_tokens() -> None:
    assert _is_garbage_query("query1") is True
    assert _is_garbage_query("q2") is True
    assert _is_garbage_query("search query 3") is True
    assert _is_garbage_query("...") is True
    assert _is_garbage_query("…") is True
    assert _is_garbage_query("[Regenerate with All equipment / sop]") is True
    assert _is_garbage_query("ashing rate too low troubleshooting") is False
