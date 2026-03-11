from __future__ import annotations

from ai_workbench.core.layout_engine import compute_stack_rows, layout_mode


def test_compute_stack_rows_focus_resize_invariants() -> None:
    total_rows = 50
    rows = compute_stack_rows(
        total_rows,
        pane_count=3,
        active_index=1,
        note_open=False,
        compact_mode=False,
        focus_resize=True,
    )
    body = total_rows - 4 - 2
    assert len(rows) == 3
    assert sum(rows) == body
    assert rows[1] >= 12
    assert min(rows[0], rows[2]) >= 3


def test_compute_stack_rows_compact_mode_small_terminal() -> None:
    total_rows = 24
    rows = compute_stack_rows(
        total_rows,
        pane_count=4,
        active_index=0,
        note_open=False,
        compact_mode=True,
        focus_resize=True,
    )
    body = total_rows - 4 - 2
    assert len(rows) == 4
    assert sum(rows) == body
    assert all(row >= 2 for row in rows)


def test_layout_mode_thresholds() -> None:
    assert layout_mode(80) == "compact"
    assert layout_mode(120) == "portrait"
    assert layout_mode(160) == "wide"
