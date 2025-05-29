"""Wizard-style Textual TUI for the Gist Memory agent."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .config import DEFAULT_BRAIN_PATH
from .embedding_pipeline import embed_text, EmbeddingDimensionMismatchError


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
        try:
            store = JsonNpyVectorStore(str(store_path))
        except EmbeddingDimensionMismatchError:
            dim = int(embed_text(["dim"]).shape[1])
            store = JsonNpyVectorStore(str(store_path), embedding_dim=dim)
    else:
        dim = int(embed_text(["dim"]).shape[1])
        store = JsonNpyVectorStore(str(store_path), embedding_dim=dim)
    agent = Agent(store)

    class HelpScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            text = (
                "Use slash commands:\n"
                "/ingest TEXT  - add a memory\n"
                "/query TEXT   - search memories\n"
                "/beliefs      - list prototypes\n"
                "/stats        - show store stats\n"
                "/exit         - quit"
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
            self.app.push_screen(ConsoleScreen())

        def action_load(self) -> None:
            sample_dir = Path("examples/moon_landing")
            for p in sorted(sample_dir.glob("*.txt")):
                agent.add_memory(p.read_text())
            self.app.pop_screen()
            self.app.push_screen(ConsoleScreen())

    class IngestScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Paste text and press Enter", id="hint")
            yield Input(id="ingest")
            yield Static("File path (Enter to ingest)", id="filehint")
            yield Input(id="file")
            yield TextLog(highlight=False, id="log")
            yield Footer()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "file":
                path = Path(event.value).expanduser()
                try:
                    text = path.read_text()
                except Exception as exc:  # pragma: no cover - runtime error path
                    log = self.query_one("#log", TextLog)
                    log.write_line(f"error reading file: {exc}")
                    event.input.value = ""
                    return
            else:
                text = event.value

            results = agent.add_memory(text)
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

    class ConsoleScreen(Screen):
        BINDINGS = []

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            self.log = TextLog(id="console")
            yield self.log
            yield Input(placeholder="/help for commands", id="cmd")
            yield Footer()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            cmd = event.value.strip()
            event.input.value = ""
            if cmd.startswith("/ingest "):
                text = cmd[len("/ingest ") :]
                results = agent.add_memory(text)
                for res in results:
                    if res.get("spawned"):
                        msg = f"spawned prototype {res['prototype_id']}"
                    else:
                        msg = f"added to {res['prototype_id']}"
                    self.log.write_line(msg)
            elif cmd.startswith("/query "):
                q = cmd[len("/query ") :]
                res = agent.query(q, top_k_prototypes=3, top_k_memories=3)
                for p in res.get("prototypes", []):
                    self.log.write_line(f"{p['sim']:.2f} {p['summary']}")
                for m in res.get("memories", []):
                    self.log.write_line(f"  {m['text']}")
            elif cmd == "/stats":
                usage = _disk_usage(store_path)
                self.log.write_line(f"disk: {usage} bytes")
                self.log.write_line(f"memories: {len(store.memories)}")
                self.log.write_line(f"prototypes: {len(store.prototypes)}")
            elif cmd == "/beliefs":
                self.app.push_screen(BeliefScreen())
            elif cmd in ("/exit", "/quit"):
                self.app.push_screen(ExitScreen())
            elif cmd in ("/help", "/?"):
                self.log.write_line("/ingest TEXT - add memory")
                self.log.write_line("/query TEXT  - search")
                self.log.write_line("/beliefs     - list prototypes")
                self.log.write_line("/stats       - show stats")
                self.log.write_line("/exit        - quit")
            elif cmd:
                self.log.write_line(f"unknown command: {cmd}")

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
        BINDINGS = [("q", "push_screen('exit')", "Quit")]

        SCREENS = {
            "help": HelpScreen,
            "console": ConsoleScreen,
            "beliefs": BeliefScreen,
            "stats": StatsScreen,
            "exit": ExitScreen,
        }

        def on_mount(self) -> None:
            self.push_screen(WelcomeScreen())

        def on_screen_resume(self, event: Screen.Resume) -> None:  # pragma: no cover
            if isinstance(event.screen, ExitScreen):
                return
            store.save()

    WizardApp().run()


__all__ = ["run_tui"]
