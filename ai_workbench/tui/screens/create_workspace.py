from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static


class CreateWorkspaceModal(ModalScreen[dict[str, str] | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, templates: list[str]) -> None:
        super().__init__()
        self._templates = templates

    def compose(self) -> ComposeResult:
        template_options = [(template, template) for template in self._templates]
        default_template = self._templates[0] if self._templates else "triple-agent"
        with Vertical(id="create-workspace-modal"):
            yield Static("Create Workspace", id="create-workspace-title")
            yield Input(placeholder="Workspace name", id="workspace-name")
            yield Input(placeholder="Project path (optional)", id="workspace-path")
            yield Select(
                template_options, value=default_template, id="workspace-template"
            )
            yield Button("Create", id="create-submit", variant="primary")
            yield Button("Cancel", id="create-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-cancel":
            self.dismiss(None)
            return

        name = self.query_one("#workspace-name", Input).value.strip()
        project_path = self.query_one("#workspace-path", Input).value.strip()
        template_widget = self.query_one("#workspace-template", Select)
        template_value = template_widget.value
        template = (
            str(template_value)
            if template_value is not Select.BLANK
            else "triple-agent"
        )

        if not name:
            self.app.notify("Workspace name is required", severity="error")
            return

        self.dismiss(
            {
                "name": name,
                "project_path": project_path,
                "template": template,
            }
        )
