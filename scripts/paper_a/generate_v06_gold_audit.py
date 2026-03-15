from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUT_JSON_PATH = ROOT / "data/paper_a/gold_verification_report.json"
OUT_MD_PATH = (
    ROOT / "docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md"
)


def main() -> None:
    report = {
        "meta": {
            "date": "2026-03-14",
            "source": "manual_inspection_snapshot_from_full_experiment_results",
            "source_doc": "docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md",
            "sampling_note": "75 sampled queries, 337 query-doc pairs manually inspected",
            "limitation": "Raw pair-level audit annotations are not yet stored in-repo; this report packages the documented aggregate results.",
        },
        "sample": {
            "queries": 75,
            "pairs_total": 337,
            "strict_pairs": 177,
            "loose_pairs": 160,
        },
        "overall": {
            "strict_precision": 0.972,
            "strict_confirmed": 172,
            "strict_partial": 5,
            "strict_false_positive": 0,
            "loose_recall": 1.0,
            "loose_confirmed": 160,
        },
        "by_scope_observability": {
            "explicit_device": {
                "estimated_query_sample": 56,
                "strict_precision": 0.993,
                "partial_rate": 0.007,
                "false_positive_rate": 0.0,
            },
            "explicit_equip": {
                "estimated_query_sample": 19,
                "strict_precision": 0.84,
                "partial_rate": 0.16,
                "false_positive_rate": 0.0,
            },
        },
        "interpretation": {
            "trusted_claims": [
                "Strict gold labels are highly reliable for explicit_device queries.",
                "Loose gold did not miss any verified relevant document in the sampled pairs.",
                "False-positive gold labels were not observed in the sampled pairs.",
            ],
            "caveats": [
                "explicit_equip strict gold is weaker than explicit_device and should be reported separately.",
                "This audit is a sampled validation, not a full-dataset rejudging pass.",
                "Because raw pair annotations are absent, confidence intervals are not yet computed in this packaged report.",
            ],
        },
    }

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# v0.6 Gold Audit (2026-03-14)",
        "",
        "Date: 2026-03-14",
        "Status: packaged from the documented manual-inspection snapshot in `2026-03-14_full_experiment_results.md`",
        "",
        "## Inputs",
        "",
        "- Source summary: `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md`",
        "- Output JSON: `data/paper_a/gold_verification_report.json`",
        "",
        "## Summary",
        "",
        "- Sampled queries: 75",
        "- Verified query-doc pairs: 337",
        "- Strict pairs: 177",
        "- Loose-only pairs: 160",
        "- Strict precision: 97.2% (172/177)",
        "- Partially relevant in strict set: 2.8% (5/177)",
        "- False positives in strict set: 0.0% (0/177)",
        "- Loose recall on sampled relevant pairs: 100.0% (160/160)",
        "",
        "## By scope_observability",
        "",
        "- explicit_device: strict precision 99.3%, partial 0.7%, false positive 0.0%",
        "- explicit_equip: strict precision 84.0%, partial 16.0%, false positive 0.0%",
        "",
        "## Interpretation",
        "",
        "- The sampled audit strongly supports using the auto-generated v0.6 gold labels for evaluation, especially on explicit_device queries.",
        "- explicit_equip remains the weaker slice and should be reported with its own caveat rather than pooled into a single precision claim.",
        "- This packaged report preserves the currently documented aggregate results, but it is not a replacement for a future raw pair-level audit artifact.",
    ]

    OUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved gold verification report to: {OUT_JSON_PATH}")
    print(f"Saved gold audit evidence to: {OUT_MD_PATH}")


if __name__ == "__main__":
    main()
