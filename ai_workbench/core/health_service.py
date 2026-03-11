from __future__ import annotations

from ai_workbench.models import (
    PaneActivity,
    PaneLifecycle,
    PaneRuntime,
    Workspace,
    WorkspaceStatus,
)

from .tmux_adapter import TmuxAdapter


class HealthService:
    def __init__(self, *, tmux: TmuxAdapter) -> None:
        self.tmux = tmux

    def assess_workspace(
        self, workspace: Workspace
    ) -> tuple[WorkspaceStatus, dict[str, PaneRuntime]]:
        session_name = workspace.session_name
        if not self.tmux.has_session(session_name):
            discovered = self.tmux.find_session_by_workspace_id(workspace.id)
            if discovered:
                session_name = discovered
            else:
                return WorkspaceStatus.MISSING, self._missing_runtimes(workspace)

        panes = self.tmux.list_panes(session_name)
        panes_by_role = {pane.role: pane for pane in panes if pane.role}

        runtimes: dict[str, PaneRuntime] = {}
        required_roles = [profile.role for profile in workspace.pane_profiles]
        missing_roles = [role for role in required_roles if role not in panes_by_role]
        has_dead = False

        for role in required_roles:
            pane = panes_by_role.get(role)
            if pane is None:
                runtimes[role] = PaneRuntime(
                    role=role,
                    lifecycle=PaneLifecycle.FAILED,
                    activity=PaneActivity.ERROR,
                    pane_dead=True,
                )
                continue

            lifecycle = PaneLifecycle.CRASHED if pane.dead else PaneLifecycle.LIVE
            activity = PaneActivity.ERROR if pane.dead else PaneActivity.IDLE
            has_dead = has_dead or pane.dead
            runtimes[role] = PaneRuntime(
                role=role,
                pane_id=pane.pane_id,
                lifecycle=lifecycle,
                activity=activity,
                pane_dead=pane.dead,
                exit_status=pane.dead_status,
            )

        if missing_roles or has_dead:
            return WorkspaceStatus.DEGRADED, runtimes

        attached = 0
        for session in self.tmux.list_sessions():
            if session.name == session_name:
                attached = session.attached_clients
                break

        if attached == 0:
            return WorkspaceStatus.DETACHED, runtimes
        return WorkspaceStatus.RUNNING, runtimes

    def _missing_runtimes(self, workspace: Workspace) -> dict[str, PaneRuntime]:
        runtimes: dict[str, PaneRuntime] = {}
        for profile in workspace.pane_profiles:
            runtimes[profile.role] = PaneRuntime(
                role=profile.role,
                lifecycle=PaneLifecycle.FAILED,
                activity=PaneActivity.ERROR,
                pane_dead=True,
            )
        return runtimes
