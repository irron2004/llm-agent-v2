from __future__ import annotations

import os

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from ai_workbench.core.health_service import HealthService
from ai_workbench.core.layout_engine import compute_stack_rows, layout_mode
from ai_workbench.core.preview_service import PreviewService
from ai_workbench.core.restore_service import RestoreService
from ai_workbench.core.workspace_service import WorkspaceService
from ai_workbench.models import (
    PaneActivity,
    PaneLifecycle,
    PaneRuntime,
    Workspace,
    WorkspaceStatus,
)
from ai_workbench.storage.note_repo import NoteRepository
from ai_workbench.templates import list_templates
from ai_workbench.tui.screens.create_workspace import CreateWorkspaceModal
from ai_workbench.tui.screens.note_modal import NoteModal
from ai_workbench.tui.screens.restore_modal import RestoreModal
from ai_workbench.tui.widgets.note_drawer import NoteDrawer
from ai_workbench.tui.widgets.pane_preview import PanePreview


class DashboardScreen(Screen[None]):
    BINDINGS = [
        ("tab", "next_workspace", "Next workspace"),
        ("shift+tab", "prev_workspace", "Previous workspace"),
        ("h", "prev_workspace", "Previous workspace"),
        ("l", "next_workspace", "Next workspace"),
        ("j", "next_pane", "Next pane"),
        ("k", "prev_pane", "Previous pane"),
        ("enter", "attach_workspace", "Attach"),
        ("m", "toggle_note", "Toggle note"),
        ("e", "toggle_note_edit", "Edit note"),
        ("ctrl+s", "save_note", "Save note"),
        ("/", "focus_search", "Search"),
        ("n", "new_workspace", "New workspace"),
        Binding("t", "add_terminal", "Add terminal", priority=True),
        ("r", "restore_workspace", "Restore"),
        ("R", "restart_pane", "Restart pane"),
        ("x", "archive_workspace", "Archive workspace"),
        ("z", "toggle_focus_resize", "Focus resize"),
        ("?", "help_overlay", "Help"),
        ("escape", "close_overlay", "Close"),
        Binding("ctrl+c", "quit_app", "Quit", priority=True),
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
        ("q", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        *,
        workspace_service: WorkspaceService,
        health_service: HealthService,
        preview_service: PreviewService,
        restore_service: RestoreService,
        note_repo: NoteRepository,
    ) -> None:
        super().__init__()
        self.workspace_service = workspace_service
        self.health_service = health_service
        self.preview_service = preview_service
        self.restore_service = restore_service
        self.note_repo = note_repo

        self.workspaces: list[Workspace] = []
        self.workspace_index = 0
        self.pane_index = 0
        self.search_query = ""
        self.focus_resize = True
        self.note_open = False
        self.note_edit = False
        self.compact_mode = False
        self.current_runtimes = {}
        self._refreshing = False
        self._preview_widgets: dict[str, PanePreview] = {}
        self._preview_role_order: tuple[str, ...] = ()
        self._tabs_cache = ""
        self._summary_cache = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="dashboard-root"):
            yield Static("No Workspaces", id="workspace-tabs")
            yield Input(
                placeholder="Search workspace",
                id="workspace-search",
                disabled=True,
            )
            yield Static("No workspace", id="workspace-summary")
            yield Vertical(id="preview-stack")
            yield NoteDrawer(id="note-drawer")
            yield Footer()

    async def on_mount(self) -> None:
        drawer = self.query_one("#note-drawer", NoteDrawer)
        drawer.display = False
        await self.refresh_from_services()
        self.set_focus(None)
        self.set_interval(1.0, self.refresh_from_services)

    async def refresh_from_services(self) -> None:
        if self.app.screen is not self:
            return
        if self._refreshing:
            return
        self._refreshing = True
        try:
            await self._refresh_impl()
        finally:
            self._refreshing = False

    async def _refresh_impl(self) -> None:
        self.compact_mode = layout_mode(self.size.width) == "compact"
        self.workspaces = self.workspace_service.list_workspaces(
            search=self.search_query,
            include_archived=True,
        )
        if self.workspaces:
            self.workspace_index = max(
                0, min(self.workspace_index, len(self.workspaces) - 1)
            )
        else:
            self.workspace_index = 0

        self._render_tabs()

        current = self.current_workspace
        if current is None:
            self.current_runtimes = {}
            summary_text = "No workspace. Press n to create one."
            if summary_text != self._summary_cache:
                self.query_one("#workspace-summary", Static).update(summary_text)
                self._summary_cache = summary_text
            await self._render_preview_stack()
            return

        try:
            status, runtimes = self.health_service.assess_workspace(current)
        except Exception as exc:
            status = WorkspaceStatus.ERROR
            current.last_error = str(exc)
            runtimes = {}

        if status != current.status:
            current.status = status
            current.touch()
            self.workspace_service.repo.upsert_workspace(current)

        recovery_state = self.workspace_service.classify_recovery_state(current)
        if recovery_state != current.recovery_state:
            current.recovery_state = recovery_state
            current.touch()
            self.workspace_service.repo.upsert_workspace(current)

        visible_roles = {profile.role for profile in current.pane_profiles}
        self.current_runtimes = self.preview_service.update_previews(
            current,
            runtimes,
            visible_roles=visible_roles,
            compact_mode=self.compact_mode,
        )

        self._update_summary()
        await self._render_preview_stack()
        self._sync_note_drawer()

    @property
    def current_workspace(self) -> Workspace | None:
        if not self.workspaces:
            return None
        return self.workspaces[self.workspace_index]

    @property
    def focused_role(self) -> str | None:
        workspace = self.current_workspace
        if workspace is None or not workspace.pane_profiles:
            return None
        self.pane_index = max(0, min(self.pane_index, len(workspace.pane_profiles) - 1))
        return workspace.pane_profiles[self.pane_index].role

    def _render_tabs(self) -> None:
        tabs = self.query_one("#workspace-tabs", Static)
        if not self.workspaces:
            next_text = "No Workspaces | +"
            if next_text != self._tabs_cache:
                tabs.update(next_text)
                self._tabs_cache = next_text
            return

        labels: list[str] = []
        for index, workspace in enumerate(self.workspaces):
            label = workspace.name
            if workspace.pinned:
                label = f"* {label}"
            if index == self.workspace_index:
                label = f"[{label}]"
            labels.append(label)
        labels.append("+")
        next_text = " | ".join(labels)
        if next_text != self._tabs_cache:
            tabs.update(next_text)
            self._tabs_cache = next_text

    def _update_summary(self) -> None:
        summary = self.query_one("#workspace-summary", Static)
        workspace = self.current_workspace
        if workspace is None:
            if self._summary_cache != "No workspace":
                summary.update("No workspace")
                self._summary_cache = "No workspace"
            return
        role = self.focused_role or "-"
        mode = "focus" if self.focus_resize else "equal"
        next_text = (
            f"{workspace.name} | session={workspace.session_name} | status={workspace.status.value.upper()}"
            f" | recovery={workspace.recovery_state.value.upper()}"
            f" | focus={role} | mode={mode}"
        )
        if next_text != self._summary_cache:
            summary.update(next_text)
            self._summary_cache = next_text

    async def _render_preview_stack(self) -> None:
        container = self.query_one("#preview-stack", Vertical)

        workspace = self.current_workspace
        if workspace is None:
            if self._preview_role_order:
                await container.remove_children()
                self._preview_widgets = {}
                self._preview_role_order = ()
            if not container.children:
                await container.mount(Static("No panes"))
            return

        role_order = tuple(profile.role for profile in workspace.pane_profiles)

        rows = compute_stack_rows(
            self.size.height,
            len(workspace.pane_profiles),
            self.pane_index,
            self.note_open and not self.compact_mode,
            compact_mode=self.compact_mode,
            focus_resize=self.focus_resize,
        )

        if role_order != self._preview_role_order:
            await container.remove_children()
            self._preview_widgets = {}
            for index, profile in enumerate(workspace.pane_profiles):
                runtime = self._runtime_for_role(profile.role)
                preview = PanePreview(
                    role=profile.role,
                    runtime=runtime,
                    active=index == self.pane_index,
                    compact_mode=self.compact_mode,
                )
                if index < len(rows):
                    preview.styles.height = rows[index]
                await container.mount(preview)
                self._preview_widgets[profile.role] = preview
            self._preview_role_order = role_order
            return

        for index, profile in enumerate(workspace.pane_profiles):
            runtime = self._runtime_for_role(profile.role)
            preview = self._preview_widgets.get(profile.role)
            if preview is None:
                self._preview_role_order = ()
                await self._render_preview_stack()
                return
            if index < len(rows):
                preview.styles.height = rows[index]
            preview.set_runtime(
                runtime,
                active=index == self.pane_index,
                compact_mode=self.compact_mode,
            )

    def _runtime_for_role(self, role: str) -> PaneRuntime:
        runtime = self.current_runtimes.get(role)
        if isinstance(runtime, PaneRuntime):
            return runtime
        return PaneRuntime(
            role=role,
            lifecycle=PaneLifecycle.FAILED,
            activity=PaneActivity.ERROR,
            pane_dead=True,
        )

    @on(Input.Changed, "#workspace-search")
    async def on_search_changed(self, event: Input.Changed) -> None:
        self.search_query = event.value.strip()
        self.workspace_index = 0
        await self.refresh_from_services()

    @on(Input.Submitted, "#workspace-search")
    async def on_search_submitted(self, event: Input.Submitted) -> None:
        event.input.disabled = True
        self.set_focus(None)

    async def action_next_workspace(self) -> None:
        if not self.workspaces:
            return
        self.workspace_index = (self.workspace_index + 1) % len(self.workspaces)
        self.pane_index = 0
        workspace = self.current_workspace
        if workspace is not None:
            self.workspace_service.set_active_workspace(workspace.id)
        await self.refresh_from_services()

    async def action_prev_workspace(self) -> None:
        if not self.workspaces:
            return
        self.workspace_index = (self.workspace_index - 1) % len(self.workspaces)
        self.pane_index = 0
        workspace = self.current_workspace
        if workspace is not None:
            self.workspace_service.set_active_workspace(workspace.id)
        await self.refresh_from_services()

    async def action_next_pane(self) -> None:
        workspace = self.current_workspace
        if workspace is None or not workspace.pane_profiles:
            return
        self.pane_index = (self.pane_index + 1) % len(workspace.pane_profiles)
        workspace.active_role = workspace.pane_profiles[self.pane_index].role
        self.workspace_service.update_workspace(workspace)
        await self.refresh_from_services()

    async def action_prev_pane(self) -> None:
        workspace = self.current_workspace
        if workspace is None or not workspace.pane_profiles:
            return
        self.pane_index = (self.pane_index - 1) % len(workspace.pane_profiles)
        workspace.active_role = workspace.pane_profiles[self.pane_index].role
        self.workspace_service.update_workspace(workspace)
        await self.refresh_from_services()

    def action_focus_search(self) -> None:
        search = self.query_one("#workspace-search", Input)
        search.disabled = False
        search.focus()

    async def action_toggle_focus_resize(self) -> None:
        self.focus_resize = not self.focus_resize
        await self.refresh_from_services()

    async def action_new_workspace(self) -> None:
        self.app.push_screen(
            CreateWorkspaceModal(list_templates()), self._on_create_workspace
        )

    async def action_add_terminal(self) -> None:
        workspace = self.current_workspace
        if workspace is None:
            await self.refresh_from_services()
            workspace = self.current_workspace
            if workspace is None:
                self.app.notify("Create a workspace first", severity="warning")
                return
        if workspace.status == WorkspaceStatus.ARCHIVED:
            self.app.notify(
                "Archived workspace cannot add new panes", severity="warning"
            )
            return

        existing_roles = {profile.role for profile in workspace.pane_profiles}
        base = "terminal"
        role = base
        next_index = 2
        while role in existing_roles:
            role = f"{base}-{next_index}"
            next_index += 1

        command = os.getenv("SHELL") or "bash"
        cwd = workspace.project_path
        try:
            updated, launched = self.workspace_service.add_pane_profile(
                workspace.id,
                role=role,
                command=command,
                cwd=cwd,
            )
        except Exception as exc:
            self.app.notify(f"Failed to add terminal pane: {exc}", severity="error")
            return

        self._upsert_local_workspace(updated)
        self.pane_index = len(updated.pane_profiles) - 1
        if self._refreshing:
            self._render_tabs()
            self._update_summary()
            await self._render_preview_stack()
        else:
            await self.refresh_from_services()
        if launched:
            self.app.notify(f"Added {role} pane")
        else:
            self.app.notify(
                f"Added {role} pane profile (session missing)", severity="warning"
            )

    def _on_create_workspace(self, payload: dict[str, str] | None) -> None:
        if payload is None:
            return
        try:
            created = self.workspace_service.create_workspace(
                name=payload["name"],
                template=payload["template"],
                project_path=payload.get("project_path") or None,
            )
        except Exception as exc:
            self.app.notify(f"Failed to create workspace: {exc}", severity="error")
            return

        refreshed = self.workspace_service.list_workspaces(include_archived=True)
        self.workspaces = refreshed
        for index, workspace in enumerate(refreshed):
            if workspace.id == created.id:
                self.workspace_index = index
                self.pane_index = 0
                break
        self._tabs_cache = ""
        self._summary_cache = ""
        self._queue_refresh()

    async def action_attach_workspace(self) -> None:
        workspace = self.current_workspace
        if workspace is None:
            return
        if workspace.status == WorkspaceStatus.ARCHIVED:
            self.app.notify("Archived workspace cannot be attached", severity="warning")
            return
        session_name = self.workspace_service.resolve_session_name(workspace)
        if not session_name:
            self.app.notify("Workspace session is missing", severity="warning")
            return

        focused_role = self.focused_role
        active_pane = (
            self.workspace_service.tmux.find_pane_by_role(session_name, focused_role)
            if focused_role
            else None
        )
        self.workspace_service.tmux.apply_portrait_layout(
            session_name,
            active_pane_id=active_pane.pane_id if active_pane else None,
        )

        inside_tmux = bool(os.getenv("TMUX"))

        home_session = self.workspace_service.tmux.attach_or_switch(
            session_name,
            inside_tmux=inside_tmux,
            exec_attach=not getattr(self.app, "test_mode", False),
        )
        if inside_tmux and home_session:
            self.app.notify(
                f"Switched to {session_name}. "
                f"Return: prefix+B or prefix+L (last session)",
            )

    async def action_restore_workspace(self) -> None:
        workspace = self.current_workspace
        if workspace is None:
            return
        if workspace.status not in {WorkspaceStatus.MISSING, WorkspaceStatus.DEGRADED}:
            self.app.notify("Restore is available only for missing/degraded workspaces")
            return
        self.app.push_screen(RestoreModal(), self._on_restore_action)

    def _on_restore_action(self, action: str | None) -> None:
        workspace = self.current_workspace
        if workspace is None or action is None:
            return
        if action == "open_note":
            self._queue_coroutine(self.action_toggle_note())
            return

        try:
            self.restore_service.restore_workspace(workspace.id, action=action)
        except Exception as exc:
            self.app.notify(f"Restore failed: {exc}", severity="error")
            return
        self._queue_refresh()

    async def action_restart_pane(self) -> None:
        workspace = self.current_workspace
        role = self.focused_role
        if workspace is None or role is None:
            return
        if workspace.status == WorkspaceStatus.ARCHIVED:
            self.app.notify(
                "Archived workspace cannot restart panes", severity="warning"
            )
            return
        try:
            self.restore_service.restart_role(workspace.id, role)
        except Exception as exc:
            self.app.notify(f"Pane restart failed: {exc}", severity="error")
            return
        await self.refresh_from_services()

    async def action_archive_workspace(self) -> None:
        workspace = self.current_workspace
        if workspace is None:
            return
        self.workspace_service.archive_workspace(workspace.id)
        await self.refresh_from_services()

    async def action_toggle_note(self) -> None:
        workspace = self.current_workspace
        if workspace is None:
            return

        content = self.note_repo.load_note(workspace.id)
        if self.compact_mode:
            self.app.push_screen(
                NoteModal(content=content, editable=self.note_edit),
                self._on_note_modal,
            )
            return

        drawer = self.query_one("#note-drawer", NoteDrawer)
        self.note_open = not self.note_open
        drawer.display = self.note_open
        if self.note_open:
            drawer.set_content(content)
            drawer.set_edit_mode(self.note_edit)
        await self.refresh_from_services()

    def _on_note_modal(self, payload: dict[str, str] | None) -> None:
        workspace = self.current_workspace
        if workspace is None or payload is None:
            return
        if payload.get("action") == "save":
            self.note_repo.save_note(workspace.id, payload.get("content", ""))
            self.app.notify("Note saved")

    async def action_toggle_note_edit(self) -> None:
        self.note_edit = not self.note_edit
        if self.compact_mode:
            await self.action_toggle_note()
            return
        drawer = self.query_one("#note-drawer", NoteDrawer)
        drawer.set_edit_mode(self.note_edit)
        self.app.notify("Note edit mode on" if self.note_edit else "Note edit mode off")

    async def action_save_note(self) -> None:
        workspace = self.current_workspace
        if workspace is None or self.compact_mode:
            return
        if not self.note_open:
            self.app.notify("Open note drawer first")
            return
        drawer = self.query_one("#note-drawer", NoteDrawer)
        self.note_repo.save_note(workspace.id, drawer.get_content())
        self.app.notify("Note saved")

    def _sync_note_drawer(self) -> None:
        if not self.note_open or self.compact_mode:
            return
        workspace = self.current_workspace
        if workspace is None:
            return
        drawer = self.query_one("#note-drawer", NoteDrawer)
        if not drawer.get_content().strip():
            drawer.set_content(self.note_repo.load_note(workspace.id))
        drawer.set_edit_mode(self.note_edit)

    def action_help_overlay(self) -> None:
        self.app.notify(
            "Keys: Tab h/l j/k Enter m e Ctrl+S / n t r R x z q Ctrl+C Ctrl+Q"
        )

    async def action_close_overlay(self) -> None:
        search = self.query_one("#workspace-search", Input)
        if not search.disabled:
            search.disabled = True
            self.set_focus(None)
            return
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
            return
        if self.note_open and not self.compact_mode:
            self.note_open = False
            drawer = self.query_one("#note-drawer", NoteDrawer)
            drawer.display = False
            await self.refresh_from_services()

    def action_quit_app(self) -> None:
        self.app.exit()

    def _queue_refresh(self) -> None:
        self._queue_coroutine(self.refresh_from_services())

    def _upsert_local_workspace(self, workspace: Workspace) -> None:
        for index, current in enumerate(self.workspaces):
            if current.id == workspace.id:
                self.workspaces[index] = workspace
                self.workspace_index = index
                return
        self.workspaces.append(workspace)
        self.workspace_index = len(self.workspaces) - 1

    def _queue_coroutine(self, coroutine) -> None:
        self.run_worker(coroutine, exclusive=True)
