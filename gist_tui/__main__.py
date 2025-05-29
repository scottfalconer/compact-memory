from __future__ import annotations

from pathlib import Path

from gist_memory.agent import Agent
from gist_memory.json_npy_store import JsonNpyVectorStore


def main(path: str = "brain") -> None:
    """Run the simple Gist Memory TUI."""
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Header, Footer, Input
        try:
            from textual.widgets import TextLog  # type: ignore
        except Exception:
            from textual.widgets import Log as TextLog  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("textual is required for gist-run") from exc

    store_path = Path(path)
    store = JsonNpyVectorStore(path=str(store_path))
    agent = Agent(store)

    class GistApp(App):
        CSS_PATH = None
        BINDINGS = [
            ("ctrl+i", "show_ingest", "Ingest"),
            ("ctrl+q", "show_query", "Query"),
            ("q", "quit", "Quit"),
        ]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            self.ingest = Input(placeholder="add text and press enter", id="ingest")
            self.query = Input(placeholder="ask a question and press enter", id="query", classes="hidden")
            self.log = TextLog()
            yield Header()
            yield self.ingest
            yield self.query
            yield self.log
            yield Footer()

        def action_show_ingest(self) -> None:
            self.ingest.remove_class("hidden")
            self.query.add_class("hidden")
            self.ingest.focus()

        def action_show_query(self) -> None:
            self.query.remove_class("hidden")
            self.ingest.add_class("hidden")
            self.query.focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "ingest":
                res = agent.add_memory(event.value)
                self.log.write_line(f"added {len(res)} chunk(s)")
                event.input.value = ""
            else:
                res = agent.query(event.value, top_k_prototypes=3, top_k_memories=3)
                for p in res["prototypes"]:
                    self.log.write_line(f"{p['sim']:.2f} {p['summary']}")
                for m in res["memories"]:
                    self.log.write_line(f"  {m['text']}")
                event.input.value = ""

    GistApp().run()


if __name__ == "__main__":
    main()
