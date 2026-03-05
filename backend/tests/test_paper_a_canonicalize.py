from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a.canonicalize import (
    canonicalize_device_name,
    compact_key,
    doc_id_variant_batch_sop,
    doc_id_variant_vlm,
    normalize_doc_type_es,
)


def test_compact_key_matches_langgraph_agent_compact_text_semantics() -> None:
    assert compact_key(None) == ""
    assert compact_key("A B-C_D.E/F") == "abcdef"
    assert compact_key(" SUPRA XP ") == "supraxp"


def test_doc_id_variant_vlm_matches_generate_doc_id_semantics() -> None:
    assert doc_id_variant_vlm("SOP (한글) v1.0__X") == "sop_한글_v1_0_x"
    assert doc_id_variant_vlm("...") == "doc"


def test_doc_id_variant_batch_sop_matches_batch_pipeline_semantics() -> None:
    assert doc_id_variant_batch_sop("SOP (한글) v1.0__X") == "sop_v1_0_x"
    assert doc_id_variant_batch_sop("  __A  ") == "_a_"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("sop_pdf", "sop"),
        ("sop_pptx", "sop"),
        ("SOP", "sop"),
        ("Global_SOP", "sop"),
        ("Troubleshooting Guide", "ts"),
        ("trouble_shooting_guide", "ts"),
        ("setup_manual", "setup"),
        ("Installation Manual", "setup"),
        ("myservice", "myservice"),
        ("maintenance", "gcb"),
    ],
)
def test_normalize_doc_type_es_maps_to_es_doc_type_keys(value: str, expected: str) -> None:
    assert normalize_doc_type_es(value) == expected


def test_canonicalize_device_name_returns_matching_candidate() -> None:
    candidates = ["SUPRA XP", "ALPHA-1"]

    assert canonicalize_device_name("supra_xp", candidates) == "SUPRA XP"
    assert canonicalize_device_name("SUPRA.XP", candidates) == "SUPRA XP"
    assert canonicalize_device_name("alpha 1", candidates) == "ALPHA-1"


def test_canonicalize_device_name_returns_none_on_miss() -> None:
    assert canonicalize_device_name("NOPE", ["SUPRA XP"]) is None
