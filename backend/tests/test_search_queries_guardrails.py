from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.langgraph_agent import validate_search_queries


def test_validate_search_queries_rejects_numeric_and_unit_injection() -> None:
    state = {
        "query": "chamber pressure alarm troubleshooting",
        "query_en": "chamber pressure alarm troubleshooting",
    }
    generated = [
        "chamber pressure alarm troubleshooting",
        "chamber pressure 50 psi alarm",
        "chamber pressure alarm 12 bar",
        "chamber pressure alarm rpm tuning",
        "chamber pressure alarm checklist",
    ]

    result = validate_search_queries(state, generated)

    assert result["search_queries"] == [
        "chamber pressure alarm troubleshooting",
        "chamber pressure alarm checklist",
    ]
    assert result["guardrail_dropped_numeric"] == 3
    assert result["guardrail_dropped_anchor"] == 0
    assert result["guardrail_final_count"] == 2


def test_validate_search_queries_rejects_anchor_drift() -> None:
    state = {
        "query": "fcip source replacement power calibration procedure",
        "query_en": "fcip source replacement power calibration procedure",
    }
    generated = [
        "football world cup final schedule",
        "fcip source replacement guide",
        "fcip source replacement power calibration method",
        "FCIP source replacement power calibration method",
    ]

    result = validate_search_queries(state, generated)

    assert result["search_queries"] == [
        "fcip source replacement power calibration procedure",
        "fcip source replacement power calibration method",
    ]
    assert result["guardrail_dropped_numeric"] == 0
    assert result["guardrail_dropped_anchor"] == 2
    assert result["guardrail_final_count"] == 2
