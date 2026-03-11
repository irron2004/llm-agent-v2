from __future__ import annotations

from ai_workbench.models import Workspace

from .workspace_service import WorkspaceService


class RestoreService:
    def __init__(self, *, workspace_service: WorkspaceService) -> None:
        self.workspace_service = workspace_service

    def restore_workspace(
        self, workspace_id: str, *, action: str = "recreate"
    ) -> Workspace:
        if action == "archive":
            return self.workspace_service.archive_workspace(workspace_id)
        if action == "open_note":
            return self.workspace_service.require_workspace(workspace_id)
        return self.workspace_service.recreate_workspace(workspace_id)

    def restart_role(self, workspace_id: str, role: str) -> Workspace:
        return self.workspace_service.restart_role(workspace_id, role)
