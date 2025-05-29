"""Wizard-style Textual TUI for the Gist Memory agent."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .config import DEFAULT_BRAIN_PATH


# ---------------------------------------------------------------------------

def _disk_usage(path: Path) -> int:
    """Return total size of files under ``path`` in bytes."""
    size = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                size += fp.stat().st_size
            except OSError:
                pass
    return size


# ---------------------------------------------------------------------------

def run_tui(path: str = DEFAULT_BRAIN_PATH) -> None:
    """Launch the Textual wizard."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container
        from textual.screen import Screen
        from textual.widgets import Header, Footer, Static, Input, DataTable
        try:  # Textual 0.x
            from textual.widgets import TextLog  # type: ignore
        except Exception:  # pragma: no cover - Textual >=1.0 renamed the widget
            from textual.widgets import Log as TextLog  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Textual is required for the TUI") from exc

    store_path = Path(path)
    meta_exists = (store_path / "meta.yaml").exists()
    if meta_exists:
        store = JsonNpyVectorStore(str(store_path))
    else:
        dim = int(embed_text(["dim"]).shape[1])
        store = JsonNpyVectorStore(str(store_path), embedding_dim=dim)
    agent = Agent(store)

    class HelpScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            text = (
                "F2 Ingest  F3 Beliefs  F4 Query  F5 Stats  Q Quit\n"
                "Navigate with arrow keys. Press Enter to select."
            )
            yield Header()
            yield Static(text, id="help")
            yield Footer()

    class WelcomeScreen(Screen):
        BINDINGS = [
            ("c", "create", "Create"),
            ("l", "load", "Load"),
        ]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static(
                "Welcome to Gist Memory\n"
                "Press [C] to create new brain, [L] to load sample.",
                id="welcome",
            )
            yield Footer()

        def action_create(self) -> None:
            self.app.pop_screen()
            self.app.push_screen(IngestScreen())

        def action_load(self) -> None:
            sample_dir = Path("examples/moon_landing")
            for p in sorted(sample_dir.glob("*.txt")):
                agent.add_memory(p.read_text())
            self.app.pop_screen()
            self.app.push_screen(BeliefScreen())

    class IngestScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Paste text and press Enter", id="hint")
            yield Input(id="ingest")
            yield TextLog(highlight=False, id="log")
            yield Footer()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            results = agent.add_memory(event.value)
            log = self.query_one("#log", TextLog)
            for res in results:
                if res.get("spawned"):
                    msg = f"spawned prototype {res['prototype_id']}"
                else:
                    msg = f"added to {res['prototype_id']}"
                log.write_line(msg)
            event.input.value = ""

    class BeliefScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            table = DataTable(id="tbl")
            table.add_columns("id", "strength", "summary")
            for p in store.prototypes:
                table.add_row(p.prototype_id[:8], str(p.strength), p.summary_text)
            yield Header()
            yield table
            yield Footer()

        def on_data_table_row_highlighted(
            self, event: DataTable.RowHighlighted
        ) -> None:
            idx = event.row_key
            if idx is None:
                return
            proto = store.prototypes[int(idx)]
            self.app.context["selected_proto"] = proto.prototype_id

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            idx = event.row_key
            proto = store.prototypes[int(idx)]
            mems = list(
                m.raw_text for m in store.memories if m.memory_id in proto.constituent_memory_ids
            )[:3]
            self.app.push_screen(DetailScreen(mems))

    class DetailScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def __init__(self, mems: Iterable[str]) -> None:
            super().__init__()
            self._mems = list(mems)

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            for m in self._mems:
                yield Static(m, classes="mem")
            yield Footer()

    class QueryScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Ask a question and press Enter", id="hint")
            yield Input(id="query")
            yield TextLog(id="answers")
            yield Footer()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            res = agent.query(event.value, top_k_prototypes=3, top_k_memories=3)
            log = self.query_one("#answers", TextLog)
            log.clear()
            for p in res.get("prototypes", []):
                log.write_line(f"{p['sim']:.2f} {p['summary']}")
            for m in res.get("memories", []):
                log.write_line(f"  {m['text']}")
            event.input.value = ""

    class StatsScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            table = DataTable(id="stats")
            table.add_columns("key", "value")
            usage = _disk_usage(store_path)
            table.add_row("disk", f"{usage} bytes")
            table.add_row("memories", str(len(store.memories)))
            table.add_row("prototypes", str(len(store.prototypes)))
            table.add_row("tau", str(agent.similarity_threshold))
            table.add_row("updated", store.meta.get("updated_at", ""))
            yield Header()
            yield table
            yield Footer()

    class ExitScreen(Screen):
        BINDINGS = [
            ("y", "yes", "Yes"),
            ("n", "no", "No"),
        ]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Save brain as zip? [Y/N]", id="exit")
            yield Footer()

        def action_yes(self) -> None:
            # simple zip using shutil.make_archive
            import shutil

            zip_path = store_path.with_suffix(".zip")
            archive = shutil.make_archive(zip_path.stem, "zip", root_dir=store_path)
            Path(archive).replace(zip_path)
            self.app.exit()

        def action_no(self) -> None:
            self.app.exit()

    class WizardApp(App):
        CSS_PATH = None
        BINDINGS = [
            ("f1", "push_screen('help')", "Help"),
            ("f2", "push_screen('ingest')", "Ingest"),
            ("f3", "push_screen('beliefs')", "Beliefs"),
            ("f4", "push_screen('query')", "Query"),
            ("f5", "push_screen('stats')", "Stats"),
            ("q", "push_screen('exit')", "Quit"),
        ]

        SCREENS = {
            "help": HelpScreen(),
            "ingest": IngestScreen(),
            "beliefs": BeliefScreen(),
            "query": QueryScreen(),
            "stats": StatsScreen(),
            "exit": ExitScreen(),
        }

        def on_mount(self) -> None:
            self.push_screen(WelcomeScreen())

        def on_screen_resume(self, event: Screen.Resume) -> None:  # pragma: no cover
            if isinstance(event.screen, ExitScreen):
                return
            store.save()

    WizardApp().run()


__all__ = ["run_tui"]
