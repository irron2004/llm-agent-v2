from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class RestoreModal(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-modal"):
            yield Static("Restore Workspace", id="restore-title")
            yield Button("Recreate Session", id="restore-recreate", variant="primary")
            yield Button("Open Note", id="restore-note")
            yield Button("Archive Workspace", id="restore-archive")
            yield Button("Cancel", id="restore-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "restore-cancel":
            self.dismiss(None)
            return
        if event.button.id == "restore-recreate":
            self.dismiss("recreate")
            return
        if event.button.id == "restore-note":
            self.dismiss("open_note")
            return
        if event.button.id == "restore-archive":
            self.dismiss("archive")
