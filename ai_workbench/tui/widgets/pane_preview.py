from __future__ import annotations

from datetime import UTC, datetime

from textual.widgets import Static

from ai_workbench.models import PaneRuntime


def _time_ago(iso_time: str | None) -> str:
    if not iso_time:
        return "-"
    try:
        parsed = datetime.fromisoformat(iso_time)
    except ValueError:
        return "-"
    now = datetime.now(UTC)
    delta = now - parsed.astimezone(UTC)
    seconds = int(max(delta.total_seconds(), 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h"


class PanePreview(Static):
    def __init__(
        self,
        *,
        role: str,
        runtime: PaneRuntime,
        active: bool,
        compact_mode: bool,
    ) -> None:
        super().__init__()
        self.role = role
        self.runtime = runtime
        self.active = active
        self.compact_mode = compact_mode

    def on_mount(self) -> None:
        if self.active:
            self.add_class("active")
        else:
            self.add_class("slim")
        self.update(self._render_text())

    def set_runtime(
        self, runtime: PaneRuntime, *, active: bool, compact_mode: bool
    ) -> None:
        self.runtime = runtime
        self.active = active
        self.compact_mode = compact_mode
        self.set_class(active, "active")
        self.set_class(not active, "slim")
        self.update(self._render_text())

    def _render_text(self) -> str:
        lifecycle = self.runtime.lifecycle.value.upper()
        activity = self.runtime.activity.value.upper()
        seen = _time_ago(self.runtime.last_output_at)
        first = f"[{self.role}] {lifecycle} | {activity} | last {seen}"

        if self.runtime.exit_status is not None and self.runtime.pane_dead:
            first += f" | exit={self.runtime.exit_status}"

        if not self.runtime.tail_preview.strip():
            return first

        lines = [
            line for line in self.runtime.tail_preview.splitlines() if line.strip()
        ]
        if not lines:
            return first

        if self.compact_mode:
            return f"{first}\n{lines[-1]}"

        if self.active:
            snippet = "\n".join(lines[-3:])
        else:
            snippet = "\n".join(lines[-2:])
        return f"{first}\n{snippet}"
