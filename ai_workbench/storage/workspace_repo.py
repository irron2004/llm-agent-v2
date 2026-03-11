from __future__ import annotations

import json
import logging
from pathlib import Path

from ai_workbench.config import (
    WorkbenchPaths,
    ensure_workbench_dirs,
    get_workbench_paths,
)
from ai_workbench.models import Workspace, WorkspaceStore

from .fileio import FileLock, atomic_write_json, backup_file

logger = logging.getLogger(__name__)


class WorkspaceRepository:
    def __init__(self, paths: WorkbenchPaths | None = None) -> None:
        resolved = ensure_workbench_dirs(paths or get_workbench_paths())
        self.paths = resolved
        self._warnings: list[str] = []

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def load_store(self) -> WorkspaceStore:
        self._warnings = []
        if not self.paths.workspaces.exists():
            return WorkspaceStore()

        loaded = self._load_from_path(self.paths.workspaces)
        if loaded is not None:
            return loaded

        backup_path = self.paths.backups_dir / "workspaces.json.bak"
        recovered = self._load_from_path(backup_path)
        if recovered is not None:
            warning = "workspaces.json was invalid, restored from backup"
            self._warnings.append(warning)
            logger.warning(warning)
            self.save_store(recovered)
            return recovered

        warning = "workspaces.json and backup were invalid, starting fresh"
        self._warnings.append(warning)
        logger.warning(warning)
        return WorkspaceStore()

    def save_store(self, store: WorkspaceStore) -> None:
        with FileLock(self.paths.lock_file):
            backup_path = self.paths.backups_dir / "workspaces.json.bak"
            backup_file(self.paths.workspaces, backup_path)
            atomic_write_json(self.paths.workspaces, store.to_dict())

    def list_workspaces(self) -> list[Workspace]:
        store = self.load_store()
        return store.workspaces

    def upsert_workspace(self, workspace: Workspace) -> None:
        store = self.load_store()
        updated: list[Workspace] = []
        replaced = False
        for current in store.workspaces:
            if current.id == workspace.id:
                updated.append(workspace)
                replaced = True
            else:
                updated.append(current)
        if not replaced:
            updated.append(workspace)
        store.workspaces = updated
        if store.active_workspace_id is None:
            store.active_workspace_id = workspace.id
        self.save_store(store)

    def remove_workspace(self, workspace_id: str) -> None:
        store = self.load_store()
        store.workspaces = [
            workspace for workspace in store.workspaces if workspace.id != workspace_id
        ]
        if store.active_workspace_id == workspace_id:
            store.active_workspace_id = (
                store.workspaces[0].id if store.workspaces else None
            )
        self.save_store(store)

    def set_active_workspace(self, workspace_id: str | None) -> None:
        store = self.load_store()
        store.active_workspace_id = workspace_id
        self.save_store(store)

    def get_active_workspace_id(self) -> str | None:
        store = self.load_store()
        return store.active_workspace_id

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        store = self.load_store()
        for workspace in store.workspaces:
            if workspace.id == workspace_id:
                return workspace
        return None

    def to_json_summary(self) -> dict[str, object]:
        store = self.load_store()
        return {
            "schema_version": store.schema_version,
            "active_workspace_id": store.active_workspace_id,
            "workspaces": [workspace.to_dict() for workspace in store.workspaces],
            "warnings": self.warnings,
        }

    def _load_from_path(self, path: Path) -> WorkspaceStore | None:
        try:
            payload = path.read_text(encoding="utf-8")
            data = json.loads(payload)
            return WorkspaceStore.from_dict(data)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Failed to load workspace store from %s: %s", path, exc)
            return None
