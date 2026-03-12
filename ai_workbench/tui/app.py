from __future__ import annotations

from textual.app import App

from ai_workbench.config import ensure_workbench_dirs, get_workbench_paths
from ai_workbench.core.health_service import HealthService
from ai_workbench.core.preview_service import PreviewService
from ai_workbench.core.restore_service import RestoreService
from ai_workbench.core.tmux_adapter import TmuxAdapter
from ai_workbench.core.workspace_service import WorkspaceService
from ai_workbench.storage.note_repo import NoteRepository
from ai_workbench.storage.workspace_repo import WorkspaceRepository
from ai_workbench.tui.screens.dashboard import DashboardScreen


class WorkbenchApp(App[None]):
    TITLE = "AI Workbench"
    SUB_TITLE = "tmux workspace dashboard"

    CSS = """
    #dashboard-root {
        layout: vertical;
        height: 100%;
    }

    #workspace-tabs {
        height: 3;
        margin: 0 1;
        border: round $panel;
    }

    #workspace-toolbar {
        margin: 0 1;
        height: 1;
        color: $text-muted;
    }

    #workspace-search {
        margin: 0 1;
    }

    #workspace-summary {
        margin: 0 1;
        height: 1;
    }

    #pane-tabs {
        height: 3;
        margin: 0 1;
        border: round $panel;
    }

    #preview-stack {
        layout: vertical;
        height: 1fr;
        margin: 0 1 1 1;
    }

    PanePreview {
        border: round $panel;
        padding: 0 1;
    }

    PanePreview.active {
        border: thick $success;
    }

    PanePreview.slim {
        border: round $boost;
    }

    #note-drawer {
        height: 12;
        margin: 0 1;
        border: round $accent;
    }

    #create-workspace-modal,
    #restore-modal,
    #note-modal {
        width: 70%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
        print_tmux: bool = False,
        socket_name: str | None = None,
        tmux_config: str | None = None,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        self.test_mode = test_mode

        paths = ensure_workbench_dirs(get_workbench_paths())
        tmux = TmuxAdapter(
            dry_run=dry_run,
            print_tmux=print_tmux,
            socket_name=socket_name,
            tmux_config=tmux_config,
        )
        workspace_repo = WorkspaceRepository(paths=paths)
        note_repo = NoteRepository(paths=paths)
        workspace_service = WorkspaceService(repo=workspace_repo, tmux=tmux)
        health_service = HealthService(tmux=tmux)
        preview_service = PreviewService(tmux=tmux)
        restore_service = RestoreService(workspace_service=workspace_service)

        self._dashboard = DashboardScreen(
            workspace_service=workspace_service,
            health_service=health_service,
            preview_service=preview_service,
            restore_service=restore_service,
            note_repo=note_repo,
        )

    def on_mount(self) -> None:
        self.push_screen(self._dashboard)
