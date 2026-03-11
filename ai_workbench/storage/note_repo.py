from __future__ import annotations

from pathlib import Path

from ai_workbench.config import (
    WorkbenchPaths,
    ensure_workbench_dirs,
    get_workbench_paths,
)

from .fileio import atomic_write_text

NOTE_TEMPLATE = """# Goal

# Current Status

# Blockers

# Next Actions
- 

# Related Files
- 

# Related Branch

"""


class NoteRepository:
    def __init__(self, paths: WorkbenchPaths | None = None) -> None:
        self.paths = ensure_workbench_dirs(paths or get_workbench_paths())

    def note_path(self, workspace_id: str) -> Path:
        return self.paths.notes_dir / f"{workspace_id}.md"

    def load_note(self, workspace_id: str) -> str:
        path = self.note_path(workspace_id)
        if not path.exists():
            return NOTE_TEMPLATE
        return path.read_text(encoding="utf-8")

    def save_note(self, workspace_id: str, content: str) -> None:
        path = self.note_path(workspace_id)
        atomic_write_text(path, content)
