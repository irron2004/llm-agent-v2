from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


class NoteModal(ModalScreen[dict[str, str] | None]):
    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("ctrl+s", "save", "Save"),
    ]

    def __init__(self, *, content: str, editable: bool) -> None:
        super().__init__()
        self._content = content
        self._editable = editable

    def compose(self) -> ComposeResult:
        with Vertical(id="note-modal"):
            yield Static("Workspace Note", id="note-modal-title")
            yield TextArea(
                self._content, id="note-modal-text", read_only=not self._editable
            )
            if self._editable:
                yield Button("Save", id="note-modal-save", variant="primary")
            yield Button("Close", id="note-modal-close")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        if not self._editable:
            return
        self.dismiss(
            {
                "action": "save",
                "content": self.query_one("#note-modal-text", TextArea).text,
            }
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "note-modal-close":
            self.dismiss(None)
            return
        if event.button.id == "note-modal-save":
            self.action_save()
