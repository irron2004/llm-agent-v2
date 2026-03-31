from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from textual.widgets import Static

from ai_workbench.core.tmux_adapter import TmuxAdapter
from ai_workbench.core.workspace_service import WorkspaceService
from ai_workbench.models import WorkspaceStatus
from ai_workbench.storage.workspace_repo import WorkspaceRepository
from ai_workbench.tui.app import WorkbenchApp
from ai_workbench.tui.screens.dashboard import DashboardScreen
from ai_workbench.tui.screens.note_modal import NoteModal
from ai_workbench.tui.screens.restore_modal import RestoreModal
from ai_workbench.tui.widgets.master_detail import MasterDetail
from ai_workbench.tui.widgets.note_drawer import NoteDrawer


def _seed_workspace(name: str) -> None:
    repo = WorkspaceRepository()
    service = WorkspaceService(repo=repo, tmux=TmuxAdapter())
    service.create_workspace(name=name, template="triple-agent")


@pytest.mark.asyncio
async def test_dashboard_renders_and_focus_resize_toggle(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("Pilot Demo")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        summary = app.screen.query_one("#workspace-summary", Static)
        before = str(summary.content)
        assert "Pilot Demo" in before

        await pilot.press("z")
        await pilot.pause()
        after = str(app.screen.query_one("#workspace-summary", Static).content)
        assert "mode=focus" in before
        assert "mode=equal" in after


@pytest.mark.asyncio
async def test_pilot_keybindings_and_note_save(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("First Workspace")
    _seed_workspace("Second Workspace")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()

        await pilot.press("l")
        await pilot.pause()
        summary = str(app.screen.query_one("#workspace-summary", Static).content)
        assert "Second Workspace" in summary

        await pilot.press("j")
        await pilot.pause()
        summary_after_focus = str(
            app.screen.query_one("#workspace-summary", Static).content
        )
        assert "focus=" in summary_after_focus

        await pilot.press("m")
        await pilot.pause()
        drawer = app.screen.query_one("#note-drawer", NoteDrawer)
        drawer.set_content("# Goal\n\nSave from pilot test")

        await pilot.press("ctrl+s")
        await pilot.pause()

        dashboard = cast(DashboardScreen, app.screen)
        workspace = dashboard.current_workspace
        assert workspace is not None
        note_path = workbench_home / "notes" / f"{workspace.id}.md"
        assert note_path.exists()
        assert "Save from pilot test" in note_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_compact_mode_shows_note_modal(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("Compact Demo")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert isinstance(app.screen, NoteModal)


@pytest.mark.asyncio
async def test_restore_modal_opens_for_missing_workspace(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("Restore Demo")
    repo = WorkspaceRepository()
    data = repo.to_json_summary()
    workspaces = cast(list[dict[str, object]], data["workspaces"])
    workspace_id = cast(str, workspaces[0]["id"])
    workspace = repo.get_workspace(workspace_id)
    assert workspace is not None
    workspace.status = WorkspaceStatus.MISSING
    repo.upsert_workspace(workspace)

    app = WorkbenchApp(dry_run=False, test_mode=True)
    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        assert isinstance(app.screen, RestoreModal)


@pytest.mark.asyncio
async def test_archive_blocks_attach_action(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("Archive Demo")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    called = {"attach": False}

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        dashboard = cast(DashboardScreen, app.screen)
        workspace = dashboard.current_workspace
        assert workspace is not None
        workspace.status = WorkspaceStatus.ARCHIVED
        dashboard.workspace_service.update_workspace(workspace)
        await dashboard.refresh_from_services()

        original_attach = dashboard.workspace_service.tmux.attach_or_switch

        def _track_attach(*args, **kwargs):
            called["attach"] = True
            return original_attach(*args, **kwargs)

        dashboard.workspace_service.tmux.attach_or_switch = _track_attach

        await pilot.press("enter")
        await pilot.pause()

    assert called["attach"] is False


@pytest.mark.asyncio
async def test_add_terminal_pane_with_t_key(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("Terminal Add Demo")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        dashboard = cast(DashboardScreen, app.screen)
        workspace = dashboard.current_workspace
        assert workspace is not None
        before_count = len(workspace.pane_profiles)

        await pilot.press("t")
        await pilot.pause()

        dashboard = cast(DashboardScreen, app.screen)
        workspace = dashboard.current_workspace
        assert workspace is not None
        assert len(workspace.pane_profiles) == before_count + 1
        assert any(
            profile.role.startswith("terminal") for profile in workspace.pane_profiles
        )


@pytest.mark.asyncio
async def test_refresh_keeps_master_detail_when_roles_unchanged(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    _seed_workspace("Stable Preview Demo")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        dashboard = cast(DashboardScreen, app.screen)
        master_detail = dashboard.query_one("#master-detail", MasterDetail)
        assert master_detail is not None

        await dashboard.refresh_from_services()
        await pilot.pause()

        dashboard = cast(DashboardScreen, app.screen)
        master_detail_after = dashboard.query_one("#master-detail", MasterDetail)
        assert master_detail is master_detail_after


@pytest.mark.asyncio
async def test_add_terminal_immediately_after_create_callback(
    workbench_home: Path,
    fake_tmux_path: Path,
) -> None:
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        dashboard = cast(DashboardScreen, app.screen)

        dashboard._on_create_workspace(
            {
                "name": "Immediate Terminal Demo",
                "project_path": "",
                "template": "triple-agent",
            }
        )

        await pilot.press("t")
        await pilot.pause()

        dashboard = cast(DashboardScreen, app.screen)
        workspace = dashboard.current_workspace
        assert workspace is not None
        assert any(
            profile.role.startswith("terminal") for profile in workspace.pane_profiles
        )


@pytest.mark.asyncio
async def test_ctrl_c_binding_triggers_quit_action(
    workbench_home: Path,
    fake_tmux_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace("Quit Binding Demo")
    app = WorkbenchApp(dry_run=False, test_mode=True)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        dashboard = cast(DashboardScreen, app.screen)
        called = {"quit": False}

        def _fake_quit() -> None:
            called["quit"] = True

        monkeypatch.setattr(dashboard, "action_quit_app", _fake_quit)

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert called["quit"] is True
