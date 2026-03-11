from __future__ import annotations


def layout_mode(width: int) -> str:
    if width < 100:
        return "compact"
    if width < 140:
        return "portrait"
    return "wide"


def _distribute_evenly(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base = total // count
    remain = total % count
    rows = [base] * count
    for index in range(remain):
        rows[index] += 1
    return rows


def compute_stack_rows(
    total_rows: int,
    pane_count: int,
    active_index: int,
    note_open: bool,
    *,
    header_rows: int = 4,
    footer_rows: int = 2,
    note_rows: int = 12,
    compact_mode: bool = False,
    focus_resize: bool = True,
) -> list[int]:
    if pane_count <= 0:
        return []

    reserved_rows = header_rows + footer_rows + (note_rows if note_open else 0)
    body_rows = max(total_rows - reserved_rows, pane_count)
    active_index = max(0, min(active_index, pane_count - 1))

    if not focus_resize:
        return _distribute_evenly(body_rows, pane_count)

    slim_min = 2 if compact_mode else (4 if body_rows >= 28 else 3)
    active_min = 12

    minimum_rows = slim_min * pane_count
    if body_rows <= minimum_rows:
        return _distribute_evenly(body_rows, pane_count)

    rows = [slim_min] * pane_count
    max_active = body_rows - (pane_count - 1) * slim_min
    active_rows = int(body_rows * 0.7)
    active_rows = max(active_min, active_rows)
    active_rows = min(active_rows, max_active)
    rows[active_index] = active_rows

    used = sum(rows)
    leftover = body_rows - used

    if leftover < 0:
        return _distribute_evenly(body_rows, pane_count)

    cursor = 0
    while leftover > 0 and pane_count > 1:
        if cursor != active_index and rows[cursor] < slim_min + 2:
            rows[cursor] += 1
            leftover -= 1
        cursor = (cursor + 1) % pane_count
        if cursor == active_index and all(
            rows[idx] >= slim_min + 2
            for idx in range(pane_count)
            if idx != active_index
        ):
            break

    if leftover > 0:
        rows[active_index] += leftover

    return rows
