from __future__ import annotations

from hashlib import sha1
import re

from ai_workbench.models import (
    PaneActivity,
    PaneLifecycle,
    PaneRuntime,
    Workspace,
    now_iso,
)

from .tmux_adapter import TmuxAdapter

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class PreviewService:
    def __init__(self, *, tmux: TmuxAdapter) -> None:
        self.tmux = tmux
        self._last_hash: dict[str, str] = {}

    def update_previews(
        self,
        workspace: Workspace,
        runtimes: dict[str, PaneRuntime],
        *,
        visible_roles: set[str],
        compact_mode: bool,
    ) -> dict[str, PaneRuntime]:
        line_count = 1 if compact_mode else 3
        for role, runtime in runtimes.items():
            if role not in visible_roles:
                continue
            if not runtime.pane_id:
                continue

            captured = self.tmux.capture_pane(runtime.pane_id, lines=200)
            clean = ANSI_ESCAPE_RE.sub("", captured)
            lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
            tail = "\n".join(lines[-line_count:])

            digest = sha1(tail.encode("utf-8")).hexdigest()
            cache_key = f"{workspace.id}:{role}"
            changed = self._last_hash.get(cache_key) != digest
            self._last_hash[cache_key] = digest

            runtime.tail_preview = tail
            if runtime.lifecycle == PaneLifecycle.LIVE:
                runtime.activity = (
                    PaneActivity.STREAMING if changed else PaneActivity.IDLE
                )
            if changed:
                runtime.last_output_at = now_iso()
        return runtimes
