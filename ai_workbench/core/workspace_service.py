from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ai_workbench.models import (
    PaneProfile,
    TmuxRuntimeHandle,
    ToolResumeRef,
    Workspace,
    WorkspaceRecoveryState,
    WorkspaceStatus,
    make_session_name,
    new_workspace_id,
    now_iso,
    validate_workspace_name,
)
from ai_workbench.storage.workspace_repo import WorkspaceRepository
from ai_workbench.templates import build_template_panes

from .tmux_adapter import TmuxAdapter, TmuxCommandError, TmuxSession


class WorkspaceService:
    def __init__(
        self,
        *,
        repo: WorkspaceRepository,
        tmux: TmuxAdapter,
    ) -> None:
        self.repo = repo
        self.tmux = tmux

    def list_workspaces(
        self,
        *,
        search: str | None = None,
        include_archived: bool = True,
    ) -> list[Workspace]:
        query = (search or "").strip().lower()
        workspaces = self.repo.list_workspaces()
        if not include_archived:
            workspaces = [
                item for item in workspaces if item.status != WorkspaceStatus.ARCHIVED
            ]
        if not query:
            return workspaces
        return [item for item in workspaces if query in item.name.lower()]

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        return self.repo.get_workspace(workspace_id)

    def get_active_workspace(self) -> Workspace | None:
        active_id = self.repo.get_active_workspace_id()
        if not active_id:
            return None
        return self.repo.get_workspace(active_id)

    def set_active_workspace(
        self, workspace_id: str, *, active_role: str | None = None
    ) -> Workspace:
        workspace = self.require_workspace(workspace_id)
        if active_role:
            workspace.active_role = active_role
            workspace.touch()
            self.repo.upsert_workspace(workspace)
        self.repo.set_active_workspace(workspace_id)
        return workspace

    def create_workspace(
        self,
        *,
        name: str,
        template: str,
        project_path: str | None = None,
        command_overrides: Mapping[str, str] | None = None,
        auto_launch: bool = True,
    ) -> Workspace:
        normalized_name = validate_workspace_name(name)
        workspace_id = new_workspace_id()
        session_name = make_session_name(workspace_id)
        panes = build_template_panes(
            template,
            project_path=project_path,
            command_overrides=command_overrides,
        )

        workspace = Workspace(
            id=workspace_id,
            name=normalized_name,
            session_name=session_name,
            template=template,
            project_path=project_path,
            pane_profiles=panes,
            active_role=panes[0].role if panes else None,
        )
        workspace.transition(WorkspaceStatus.STARTING)
        self.repo.upsert_workspace(workspace)
        self.repo.set_active_workspace(workspace.id)

        if not auto_launch:
            workspace.recovery_state = self.classify_recovery_state(workspace)
            self.repo.upsert_workspace(workspace)
            return workspace

        try:
            self.launch_workspace(workspace)
            workspace.transition(WorkspaceStatus.RUNNING)
            workspace.recovery_state = WorkspaceRecoveryState.LIVE
            workspace.last_error = None
        except Exception as exc:
            workspace.status = WorkspaceStatus.ERROR
            workspace.recovery_state = self.classify_recovery_state(workspace)
            workspace.last_error = str(exc)
            workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace

    def launch_workspace(self, workspace: Workspace) -> Workspace:
        if not workspace.pane_profiles:
            raise ValueError("Workspace has no pane profiles")

        first = workspace.pane_profiles[0]
        pane_ids_by_role: dict[str, str] = {}
        first_pane = self.tmux.new_session(
            workspace.session_name,
            first.command,
            cwd=first.cwd,
        )
        pane_ids_by_role[first.role] = first_pane
        self.tmux.set_session_workspace_id(workspace.session_name, workspace.id)
        self.tmux.set_pane_role(first_pane, first.role)

        for profile in workspace.pane_profiles[1:]:
            pane_id = self.tmux.split_window(
                workspace.session_name,
                profile.command,
                cwd=profile.cwd,
            )
            pane_ids_by_role[profile.role] = pane_id
            self.tmux.set_pane_role(pane_id, profile.role)

        self.tmux.select_layout(workspace.session_name)
        self.tmux.configure_workspace_bar(workspace.session_name)
        self._apply_runtime_snapshot(workspace, pane_ids_by_role)
        workspace.active_role = first.role
        workspace.recovery_state = WorkspaceRecoveryState.LIVE
        workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace

    def resolve_session_name(self, workspace: Workspace) -> str | None:
        if self.tmux.has_session(workspace.session_name):
            self._apply_runtime_snapshot(workspace, None)
            workspace.recovery_state = WorkspaceRecoveryState.LIVE
            workspace.touch()
            self.repo.upsert_workspace(workspace)
            return workspace.session_name

        discovered = self.tmux.find_session_by_workspace_id(workspace.id)
        if discovered and discovered != workspace.session_name:
            workspace.session_name = discovered
            workspace.touch()
            self.repo.upsert_workspace(workspace)
        if discovered:
            self._apply_runtime_snapshot(workspace, None)
            workspace.recovery_state = WorkspaceRecoveryState.LIVE
            workspace.touch()
            self.repo.upsert_workspace(workspace)
        else:
            workspace.recovery_state = self.classify_recovery_state(workspace)
            workspace.touch()
            self.repo.upsert_workspace(workspace)
        return discovered

    def archive_workspace(self, workspace_id: str) -> Workspace:
        workspace = self.require_workspace(workspace_id)
        if workspace.status == WorkspaceStatus.ARCHIVED:
            return workspace

        if workspace.status in {
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.DETACHED,
            WorkspaceStatus.ERROR,
        }:
            workspace.transition(WorkspaceStatus.ARCHIVED)
        else:
            workspace.status = WorkspaceStatus.ARCHIVED
            workspace.touch()

        self.repo.upsert_workspace(workspace)
        return workspace

    def pin_workspace(self, workspace_id: str, *, pinned: bool) -> Workspace:
        workspace = self.require_workspace(workspace_id)
        workspace.pinned = pinned
        workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace

    def recreate_workspace(self, workspace_id: str) -> Workspace:
        workspace = self.require_workspace(workspace_id)
        session_name = self.resolve_session_name(workspace)
        if session_name:
            self.tmux.kill_managed_session(session_name, workspace.id)

        workspace.status = WorkspaceStatus.RESTORING
        workspace.touch()
        self.repo.upsert_workspace(workspace)

        self.launch_workspace(workspace)
        workspace.status = WorkspaceStatus.RUNNING
        workspace.recovery_state = WorkspaceRecoveryState.LIVE
        workspace.last_error = None
        workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace

    def restart_role(self, workspace_id: str, role: str) -> Workspace:
        workspace = self.require_workspace(workspace_id)
        session_name = self.resolve_session_name(workspace)
        if not session_name:
            raise ValueError("Workspace session is missing")

        profile = next(
            (item for item in workspace.pane_profiles if item.role == role), None
        )
        if profile is None:
            raise ValueError(f"Unknown role: {role}")

        pane = self.tmux.find_pane_by_role(session_name, role)
        if pane is None:
            raise ValueError(f"No pane found for role: {role}")

        self.tmux.respawn_pane(pane.pane_id, profile.command, cwd=profile.cwd)
        profile.pane_id_last_seen = pane.pane_id
        self._apply_runtime_snapshot(workspace, {role: pane.pane_id})
        workspace.recovery_state = WorkspaceRecoveryState.LIVE
        workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace

    def add_pane_profile(
        self,
        workspace_id: str,
        *,
        role: str,
        command: str,
        cwd: str | None = None,
    ) -> tuple[Workspace, bool]:
        workspace = self.require_workspace(workspace_id)
        normalized_role = role.strip()
        normalized_command = command.strip()
        if not normalized_role:
            raise ValueError("Pane role must not be empty")
        if not normalized_command:
            raise ValueError("Pane command must not be empty")

        if any(profile.role == normalized_role for profile in workspace.pane_profiles):
            raise ValueError(f"Pane role already exists: {normalized_role}")

        launched = False
        pane_id: str | None = None
        if workspace.status != WorkspaceStatus.ARCHIVED:
            session_name = self.resolve_session_name(workspace)
            if session_name:
                try:
                    pane_id = self.tmux.split_window(
                        session_name, normalized_command, cwd=cwd
                    )
                    self.tmux.set_pane_role(pane_id, normalized_role)
                    self.tmux.select_layout(session_name)
                    launched = True
                except TmuxCommandError as exc:
                    if exc.kind not in {
                        "pane_missing",
                        "session_missing",
                        "server_missing",
                    }:
                        raise
                    pane_id = self.tmux.new_session(
                        workspace.session_name,
                        normalized_command,
                        cwd=cwd,
                    )
                    self.tmux.set_session_workspace_id(
                        workspace.session_name, workspace.id
                    )
                    self.tmux.set_pane_role(pane_id, normalized_role)
                    launched = True
            else:
                pane_id = self.tmux.new_session(
                    workspace.session_name,
                    normalized_command,
                    cwd=cwd,
                )
                self.tmux.set_session_workspace_id(workspace.session_name, workspace.id)
                self.tmux.set_pane_role(pane_id, normalized_role)
                launched = True

        profile = PaneProfile(
            role=normalized_role,
            command=normalized_command,
            cwd=cwd,
            pane_id_last_seen=pane_id,
        )
        workspace.pane_profiles.append(profile)
        workspace.active_role = normalized_role
        if launched:
            workspace.status = WorkspaceStatus.RUNNING
            workspace.last_error = None
            if pane_id is not None:
                self._apply_runtime_snapshot(workspace, {normalized_role: pane_id})
            workspace.recovery_state = WorkspaceRecoveryState.LIVE
        else:
            workspace.recovery_state = self.classify_recovery_state(workspace)
        workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace, launched

    def set_pane_resume_session(
        self,
        workspace_id: str,
        *,
        role: str,
        kind: str,
        session_id: str,
    ) -> Workspace:
        workspace = self.require_workspace(workspace_id)
        profile = next(
            (item for item in workspace.pane_profiles if item.role == role), None
        )
        if profile is None:
            raise ValueError(f"Unknown role: {role}")
        profile.resume = ToolResumeRef(kind=kind, session_id=session_id)
        workspace.touch()
        self.repo.upsert_workspace(workspace)
        return workspace

    def classify_recovery_state(self, workspace: Workspace) -> WorkspaceRecoveryState:
        live_session = self._find_live_session(workspace)
        if live_session is not None:
            return WorkspaceRecoveryState.LIVE
        if not workspace.pane_profiles:
            return WorkspaceRecoveryState.BROKEN
        if any(
            profile.resume is not None and profile.resume.session_id
            for profile in workspace.pane_profiles
        ):
            return WorkspaceRecoveryState.RECOVERABLE
        return WorkspaceRecoveryState.REBUILDABLE

    def _find_live_session(self, workspace: Workspace) -> TmuxSession | None:
        for session in self.tmux.list_sessions():
            if session.workspace_id == workspace.id:
                return session
            if session.name == workspace.session_name:
                return session
        return None

    def _apply_runtime_snapshot(
        self,
        workspace: Workspace,
        pane_ids_by_role: dict[str, str] | None,
    ) -> None:
        live_session = self._find_live_session(workspace)
        if live_session is None:
            return

        workspace.tmux_runtime = TmuxRuntimeHandle(
            socket_path=live_session.socket_path,
            session_name=live_session.name,
            session_id_last_seen=live_session.session_id,
            last_seen_at=now_iso(),
        )
        if live_session.name:
            workspace.session_name = live_session.name

        role_to_pane: dict[str, str] = {}
        if pane_ids_by_role is not None:
            role_to_pane.update(pane_ids_by_role)
        else:
            try:
                panes = self.tmux.list_panes(live_session.name)
            except Exception:
                panes = []
            for pane in panes:
                if pane.role:
                    role_to_pane[pane.role] = pane.pane_id

        for profile in workspace.pane_profiles:
            pane_id = role_to_pane.get(profile.role)
            if pane_id:
                profile.pane_id_last_seen = pane_id

    def update_workspace(self, workspace: Workspace) -> Workspace:
        cloned = replace(workspace)
        cloned.touch()
        self.repo.upsert_workspace(cloned)
        return cloned

    def require_workspace(self, workspace_id: str) -> Workspace:
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            raise ValueError(f"Workspace not found: {workspace_id}")
        return workspace
