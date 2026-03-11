from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Static, TextArea


class NoteDrawer(Vertical):
    def compose(self):
        yield Static("Notes", id="note-drawer-title")
        yield TextArea("", id="note-drawer-text", read_only=True)

    def set_content(self, text: str) -> None:
        area = self.query_one("#note-drawer-text", TextArea)
        area.text = text

    def get_content(self) -> str:
        area = self.query_one("#note-drawer-text", TextArea)
        return area.text

    def set_edit_mode(self, enabled: bool) -> None:
        area = self.query_one("#note-drawer-text", TextArea)
        area.read_only = not enabled
