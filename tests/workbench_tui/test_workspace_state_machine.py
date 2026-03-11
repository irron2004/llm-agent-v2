from __future__ import annotations

from ai_workbench.core.tmux_adapter import TmuxAdapter, TmuxCommandError
from ai_workbench.core.workspace_service import WorkspaceService
from ai_workbench.models import (
    PaneProfile,
    ToolResumeRef,
    Workspace,
    WorkspaceRecoveryState,
    WorkspaceStatus,
)
from ai_workbench.storage.workspace_repo import WorkspaceRepository
from ai_workbench.templates import build_template_panes


def test_workspace_state_machine_valid_transition() -> None:
    workspace = Workspace(
        id="ws-1",
        name="Demo",
        session_name="aiwb-11111111",
        template="triple-agent",
        pane_profiles=[PaneProfile(role="claude", command="claude")],
    )
    workspace.transition(WorkspaceStatus.STARTING)
    workspace.transition(WorkspaceStatus.RUNNING)
    workspace.transition(WorkspaceStatus.DETACHED)
    workspace.transition(WorkspaceStatus.RUNNING)
    assert workspace.status == WorkspaceStatus.RUNNING


def test_workspace_state_machine_invalid_transition_rejected() -> None:
    workspace = Workspace(
        id="ws-1",
        name="Demo",
        session_name="aiwb-11111111",
        template="triple-agent",
        pane_profiles=[PaneProfile(role="claude", command="claude")],
    )
    try:
        workspace.transition(WorkspaceStatus.ARCHIVED)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid transition to raise ValueError")


def test_workspace_create_assigns_safe_session_name(workbench_home) -> None:
    repo = WorkspaceRepository()
    service = WorkspaceService(repo=repo, tmux=TmuxAdapter(dry_run=True))

    workspace = service.create_workspace(name="Demo Workspace", template="triple-agent")
    assert workspace.session_name.startswith("aiwb-")
    assert not workspace.session_name.startswith("=")
    assert workspace.status in {WorkspaceStatus.RUNNING, WorkspaceStatus.ERROR}


def test_terminal_template_uses_shell_command() -> None:
    panes = build_template_panes("terminal")
    assert len(panes) == 1
    assert panes[0].role == "terminal"
    assert panes[0].command


def test_add_pane_profile_bootstraps_session_when_missing(
    workbench_home, fake_tmux_path
) -> None:
    repo = WorkspaceRepository()
    service = WorkspaceService(repo=repo, tmux=TmuxAdapter())

    created = service.create_workspace(name="Demo Workspace", template="triple-agent")
    updated, launched = service.add_pane_profile(
        created.id,
        role="terminal",
        command="bash",
        cwd=created.project_path,
    )

    assert launched is True
    assert any(profile.role == "terminal" for profile in updated.pane_profiles)


def test_add_pane_profile_recovers_from_split_target_race(
    workbench_home, monkeypatch
) -> None:
    repo = WorkspaceRepository()
    adapter = TmuxAdapter(dry_run=True)
    service = WorkspaceService(repo=repo, tmux=adapter)
    created = service.create_workspace(name="Race Recover", template="triple-agent")

    new_session_called = {"count": 0}

    def _split_fail(*args, **kwargs):
        raise TmuxCommandError(
            "tmux command failed: can't find pane", kind="pane_missing"
        )

    def _new_session(*args, **kwargs):
        new_session_called["count"] += 1
        return "%55"

    monkeypatch.setattr(adapter, "split_window", _split_fail)
    monkeypatch.setattr(adapter, "new_session", _new_session)

    updated, launched = service.add_pane_profile(
        created.id,
        role="terminal",
        command="bash",
        cwd=created.project_path,
    )

    assert launched is True
    assert new_session_called["count"] == 1
    assert any(profile.role == "terminal" for profile in updated.pane_profiles)


def test_classify_recovery_state_variants(fake_tmux_path) -> None:
    service = WorkspaceService(repo=WorkspaceRepository(), tmux=TmuxAdapter())

    live = Workspace(
        id="ws-renamed",
        name="Live",
        session_name="renamed-session",
        template="triple-agent",
        pane_profiles=[PaneProfile(role="claude", command="claude")],
    )
    assert service.classify_recovery_state(live) == WorkspaceRecoveryState.LIVE

    recoverable = Workspace(
        id="ws-missing",
        name="Recoverable",
        session_name="missing-session",
        template="triple-agent",
        pane_profiles=[
            PaneProfile(
                role="claude",
                command="claude",
                resume=ToolResumeRef(kind="claude", session_id="sess-1"),
            )
        ],
    )
    assert (
        service.classify_recovery_state(recoverable)
        == WorkspaceRecoveryState.RECOVERABLE
    )

    rebuildable = Workspace(
        id="ws-rebuild",
        name="Rebuildable",
        session_name="missing-session-2",
        template="triple-agent",
        pane_profiles=[PaneProfile(role="claude", command="claude")],
    )
    assert (
        service.classify_recovery_state(rebuildable)
        == WorkspaceRecoveryState.REBUILDABLE
    )

    broken = Workspace(
        id="ws-broken",
        name="Broken",
        session_name="missing-session-3",
        template="terminal",
        pane_profiles=[],
    )
    assert service.classify_recovery_state(broken) == WorkspaceRecoveryState.BROKEN
