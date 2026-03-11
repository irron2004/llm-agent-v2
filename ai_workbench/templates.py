from __future__ import annotations

import os
from typing import Mapping

from .models import PaneProfile

WORKSPACE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "triple-agent": ("claude", "codex", "opencode"),
    "research": ("claude", "shell", "logs"),
    "debug": ("opencode", "shell", "logs"),
    "writing": ("claude", "shell"),
    "terminal": ("terminal",),
}

DEFAULT_ROLE_COMMANDS: dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "shell": os.getenv("SHELL") or "bash",
    "terminal": os.getenv("SHELL") or "bash",
    "logs": "tail -f /dev/null",
}


def list_templates() -> list[str]:
    return sorted(WORKSPACE_TEMPLATES.keys())


def build_template_panes(
    template: str,
    *,
    project_path: str | None = None,
    command_overrides: Mapping[str, str] | None = None,
) -> list[PaneProfile]:
    if template not in WORKSPACE_TEMPLATES:
        raise ValueError(f"Unsupported template: {template}")

    overrides = command_overrides or {}
    panes: list[PaneProfile] = []
    for role in WORKSPACE_TEMPLATES[template]:
        command = overrides.get(role, DEFAULT_ROLE_COMMANDS.get(role, role))
        panes.append(PaneProfile(role=role, command=command, cwd=project_path))
    return panes
