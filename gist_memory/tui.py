"""Simple Textual-based TUI for ingesting text files or manual input."""
from __future__ import annotations

from pathlib import Path

from .memory_creation import IdentityMemoryCreator
from .store import PrototypeStore


def run_tui() -> None:
    """Launch the Textual TUI."""
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Header, Footer, Label, Input, ListView, ListItem
        from textual.containers import Container
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Textual is required for the TUI") from exc

    class WelcomeApp(App):
        """Display a short welcome message before launching the main TUI."""

        CSS_PATH = None

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Label("Welcome to Gist Memory", id="welcome")
            yield Label(
                "Commands: ingest, query, decode, summarize, dump",
                id="commands",
            )
            yield Label("Press any key to continue", id="continue")
            yield Footer()

        def on_key(self, event) -> None:  # pragma: no cover - simple handler
            self.exit()

    class IngestApp(App):
        """Basic ingestion interface."""

        CSS_PATH = None

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Label("Select a .txt file or press i to type text", id="hint")
            files = [f for f in Path('.').glob('*.txt')]
            items = [ListItem(Label(f.name), id=str(f)) for f in files]
            yield ListView(*items, id="files")
            yield Input(placeholder="enter text", id="input", classes="hidden")
            yield Label("", id="status")
            yield Footer()

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            path = Path(event.item.id)
            text = path.read_text()
            self._ingest(text)

        def key_i(self) -> None:
            inp = self.query_one("#input", Input)
            inp.remove_class("hidden")
            inp.focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self._ingest(event.value)
            event.input.value = ""
            event.input.add_class("hidden")

        def _ingest(self, text: str) -> None:
            store = PrototypeStore()
            creator = IdentityMemoryCreator()
            for chunk in creator.create_all(text):
                mem = store.add_memory(chunk)
            status = self.query_one("#status", Label)
            status.update("Ingested text")

    WelcomeApp().run()
    IngestApp().run()


__all__ = ["run_tui"]
