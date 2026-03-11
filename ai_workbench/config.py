from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ENV_WORKBENCH_HOME = "AI_WORKBENCH_HOME"
DEFAULT_WORKBENCH_HOME = ".ai-workbench"


@dataclass(frozen=True, slots=True)
class WorkbenchPaths:
    home: Path
    workspaces: Path
    runtime: Path
    notes_dir: Path
    backups_dir: Path
    lock_file: Path


def get_workbench_paths() -> WorkbenchPaths:
    raw_home = os.getenv(ENV_WORKBENCH_HOME)
    if raw_home:
        home = Path(raw_home).expanduser().resolve()
    else:
        home = (Path.home() / DEFAULT_WORKBENCH_HOME).resolve()

    return WorkbenchPaths(
        home=home,
        workspaces=home / "workspaces.json",
        runtime=home / "runtime.json",
        notes_dir=home / "notes",
        backups_dir=home / "backups",
        lock_file=home / ".workspace.lock",
    )


def ensure_workbench_dirs(paths: WorkbenchPaths | None = None) -> WorkbenchPaths:
    resolved = paths or get_workbench_paths()
    resolved.home.mkdir(parents=True, exist_ok=True)
    resolved.notes_dir.mkdir(parents=True, exist_ok=True)
    resolved.backups_dir.mkdir(parents=True, exist_ok=True)
    return resolved
