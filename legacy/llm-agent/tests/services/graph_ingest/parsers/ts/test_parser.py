from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from services.graph_ingest.parsers.ts.parser import parse_ts_guide, validate_ts


def test_parse_ts_guide_from_markdown_sample() -> None:
    markdown = """
Trouble Shooting Guide [Trace Vacuum-Vent abnormal]

➢ A. Fast vacuum timeout
➢ B. Fast vent timeout
➢ C. Slow vent timeout
➢ D. Slow vacuum timeout

Failure symptoms - Key point
Fast vacuum timeout
A-1. GUI ▶ Check parameter setting
A-2. Leak
- Check chamber o-ring status
(Follow spec)

Fast vent timeout
B-1. GUI
- Inspect system logs

Appendix #1
|Appendix #1|Appendix #1|<br>|---|---|<br>|A|• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_PM_DEVICE NET BOARD<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_SUB UNIT_SOLENOID VALVE<br><br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_FAST VACUUM VALVE|<br>|B|• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_PNEUMATIC VALVE<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_FAST VACUUM VALVE<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_PIRANI GAUGE<br><br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_PM_DEVICE NET BOARD<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_SUB UNIT_SOLENOID VALVE|<br>|C|• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_PNEUMATIC VALVE<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_FAST VACUUM VALVE<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_PIRANI GAUGE<br><br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_PM_DEVICE NET BOARD<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_SUB UNIT_SOLENOID VALVE|<br>|D|• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_PM_DOOR VALVE<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_REP_PM_SLOW VAC VALVE<br><br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_PM_DEVICE NET BOARD<br><br>• GCB – GFP – Global SOP - Global SOP_SUPRA N_ALL_SUB UNIT_SOLENOID VALVE|<br>
"""

    structure = parse_ts_guide(markdown, ts_filename="Trace.pdf")

    assert structure["ts_id"] == "Trace Vacuum-Vent abnormal"
    assert structure["title"] == "Trouble Shooting Guide [Trace Vacuum-Vent abnormal]"

    issue_a = structure["issues"]["A"]
    assert issue_a["label"] == "Fast vacuum timeout"
    assert issue_a["step_codes"] == ["A-1", "A-2"]
    assert issue_a["steps"]["A-1. GUI"] == ["Check parameter setting"]
    assert issue_a["steps"]["A-2. Leak"] == ["Check chamber o-ring status (Follow spec)"]

    issue_b = structure["issues"]["B"]
    assert issue_b["step_codes"] == ["B-1"]
    assert issue_b["steps"]["B-1. GUI"] == ["Inspect system logs"]

    labeled_issue = structure["issue"]["Fast vacuum timeout"]
    assert "A-1. GUI" in labeled_issue

    refs_a = structure["references"]["A"]
    assert refs_a == [
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_ALL_PM_DEVICE NET BOARD",
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_ALL_SUB UNIT_SOLENOID VALVE",
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_REP_PM_FAST VACUUM VALVE",
    ]

    refs_b = structure["references"]["B"]
    assert refs_b == [
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_REP_PM_PNEUMATIC VALVE",
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_REP_PM_FAST VACUUM VALVE",
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_REP_PM_PIRANI GAUGE",
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_ALL_PM_DEVICE NET BOARD",
        "GCB - GFP - Global SOP - Global SOP_SUPRA N_ALL_SUB UNIT_SOLENOID VALVE",
    ]

    assert structure["diagnostics"]["unmapped_refs"] == []

    assert "Confidential" not in structure["cleaned_text"]
    assert structure["meta"]["source_name"] == "Trace.pdf"

    warnings = validate_ts(structure)
    assert warnings["missing_label"] == []
    assert warnings["missing_steps"] == ["C", "D"]
