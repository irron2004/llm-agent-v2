from __future__ import annotations

from scripts.evaluation.run_sop_filter_eval import (
    DEFAULT_ANSWER_PREVIEW_CHARS,
    _check_hit as filter_check_hit,
)
from scripts.evaluation.run_sop_retrieval_eval import (
    _check_hit as retrieval_check_hit,
    _normalize as retrieval_normalize,
)


def test_retrieval_normalize_handles_ampersand_and_symbols() -> None:
    raw = "global sop_supra xp_sw_all_sw installation & setting.pdf"
    assert (
        retrieval_normalize(raw) == "global_sop_supra_xp_sw_all_sw_installation_setting"
    )


def test_retrieval_check_hit_uses_best_page_for_matched_doc() -> None:
    docs = [
        {"doc_id": "global_sop_zedius_xp_rep_pm_pirani_gauge", "page": 1},
        {"doc_id": "global_sop_zedius_xp_rep_pm_pirani_gauge", "page": 22},
    ]

    hit_doc, hit_page, hit_rank = retrieval_check_hit(
        docs,
        gold_doc="Global SOP_ZEDIUS XP_REP_PM_PIRANI GAUGE.pdf",
        gold_pages="22-30",
    )

    assert hit_doc is True
    assert hit_page is True
    assert hit_rank == 1


def test_filter_check_hit_uses_best_page_for_matched_doc() -> None:
    docs = [
        {
            "doc_id": "global_sop_zedius_xp_rep_pm_pirani_gauge",
            "title": "Global SOP ZEDIUS XP REP PM PIRANI GAUGE",
            "page": 2,
            "metadata": {"source": "Global SOP_ZEDIUS XP_REP_PM_PIRANI GAUGE.pdf"},
        },
        {
            "doc_id": "global_sop_zedius_xp_rep_pm_pirani_gauge",
            "title": "Global SOP ZEDIUS XP REP PM PIRANI GAUGE",
            "page": 25,
            "metadata": {"source": "Global SOP_ZEDIUS XP_REP_PM_PIRANI GAUGE.pdf"},
        },
    ]

    hit_doc, hit_page, hit_rank, match_debug = filter_check_hit(
        docs,
        gold_doc="Global SOP_ZEDIUS XP_REP_PM_PIRANI GAUGE.pdf",
        gold_pages="22-30",
    )

    assert hit_doc is True
    assert hit_page is True
    assert hit_rank == 1
    assert match_debug["rank"] == 1


def test_filter_check_hit_matches_ampersand_normalized_doc_id() -> None:
    docs = [
        {
            "doc_id": "global_sop_supra_xp_sw_all_sw_installation_setting",
            "title": "Global SOP SUPRA XP SW ALL SW Installation Setting",
            "page": 7,
            "metadata": {
                "source": "global sop_supra xp_sw_all_sw installation setting.pdf"
            },
        }
    ]

    hit_doc, hit_page, hit_rank, _ = filter_check_hit(
        docs,
        gold_doc="global sop_supra xp_sw_all_sw installation & setting.pdf",
        gold_pages="7-9",
    )

    assert hit_doc is True
    assert hit_page is True
    assert hit_rank == 1


def test_filter_eval_default_preview_chars_is_2000() -> None:
    assert DEFAULT_ANSWER_PREVIEW_CHARS == 2000
