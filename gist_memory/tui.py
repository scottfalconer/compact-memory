"""Wizard-style Textual TUI for the Gist Memory agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .config import DEFAULT_BRAIN_PATH
from .embedding_pipeline import embed_text, EmbeddingDimensionMismatchError
from .logging_utils import configure_logging
from .talk_session import TalkSessionManager

from .memory_cues import MemoryCueRenderer



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


def _install_models(
    embed_model: str = "all-MiniLM-L6-v2", chat_model: str = "distilgpt2"
) -> str:
    """Download the default embedding and chat models."""
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        SentenceTransformer(embed_model)
        AutoTokenizer.from_pretrained(chat_model)
        AutoModelForCausalLM.from_pretrained(chat_model)
    except Exception as exc:  # pragma: no cover - network / file errors
        return f"error installing models: {exc}"

    return f"installed {embed_model} and {chat_model}"


# ---------------------------------------------------------------------------


def run_tui(path: str = DEFAULT_BRAIN_PATH) -> None:
    """Launch the Textual wizard."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container
        from textual.screen import Screen
        from textual.widgets import Header, Footer, Static, Input, DataTable
        from .autocomplete_input import TabAutocompleteInput

        try:  # Textual 0.x
            from textual.widgets import TextLog  # type: ignore
        except Exception:  # pragma: no cover - Textual >=1.0 renamed the widget
            from textual.widgets import Log as TextLog  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Textual is required for the TUI") from exc

    class StatusMixin(Screen):
        """Screen mixin providing a status bar helper."""

        def set_status(self, message: str, *, error: bool = False) -> None:
            bar = self.query_one("#status", Static)
            if error:
                bar.update(f"[red]{message}[/]", markup=True)
            else:
                bar.update(message)

    class MessageModal(StatusMixin):
        """Modal dialog for critical messages."""

        BINDINGS = [("enter", "dismiss", "OK"), ("escape", "dismiss", "OK")]
        modal = True

        def __init__(self, message: str) -> None:
            super().__init__()
            self._message = message

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static(self._message, id="modal")
            yield Static("", id="status")
            yield Footer()

        def action_dismiss(self) -> None:
            self.app.pop_screen()

    store_path = Path(path)
    meta_exists = (store_path / "meta.yaml").exists()
    if meta_exists:
        try:
            store = JsonNpyVectorStore(str(store_path))
        except Exception as exc:
            raise RuntimeError(
                f"Error: Brain data is corrupted. {exc}. "
                f"Try running gist-memory validate {store_path} for more details or restore from a backup."
            ) from exc
    else:
        try:
            dim = int(embed_text(["dim"]).shape[1])
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
        store = JsonNpyVectorStore(str(store_path), embedding_dim=dim)
    agent = Agent(store)
    talk_mgr = TalkSessionManager()
    session_id = talk_mgr.create_session([store_path])

    class HelpScreen(Screen):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            text = (
                "Use slash commands:\n"
                "/ingest TEXT      - add a memory\n"
                "/query TEXT       - search memories\n"
                "/beliefs          - list prototypes\n"
                "/stats            - show store stats\n"
                "/install-models   - download models\n"
                "/talk             - chat session\n"
                "/log PATH        - write debug log\n"
                "/exit             - quit"
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
            text = (
                "Welcome to Gist Memory\n"
                "Press [C] to create a new brain or [L] to load a sample.\n"
                "Once the console opens type /help for commands.\n"
                "Use F5 for stats and Q to quit."
            )
            yield Static(text, id="welcome")
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

    class IngestScreen(StatusMixin):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Paste text and press Enter", id="hint")
            yield Input(id="ingest")
            yield Static("File path (Enter to ingest)", id="filehint")
            yield Input(id="file")
            yield Static("", id="fileerr")
            yield TextLog(highlight=False, id="log")
            yield Static("", id="status")
            yield Footer()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.set_status("Processing...")
            event.input.disabled = True
            if event.input.id == "file":
                path = Path(event.value).expanduser()
                if not path.exists():
                    self.set_status("File does not exist.", error=True)
                    event.input.disabled = False
                    return
                try:
                    text = path.read_text()
                except Exception as exc:  # pragma: no cover - runtime error path
                    log = self.query_one("#log", TextLog)
                    log.write_line(f"Error: {exc}", style="red")
                    self.set_status(f"error reading file", error=True)
                    event.input.value = ""
                    event.input.disabled = False
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
            self.set_status("Memory ingested.")
            event.input.disabled = False

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "file":
                err = self.query_one("#fileerr", Static)
                path = Path(event.value).expanduser()
                if path.exists():
                    err.update("")
                else:
                    err.update("File does not exist.", style="red")

    class BeliefScreen(StatusMixin):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            table = DataTable(id="tbl")
            table.add_columns("id", "strength", "summary")
            yield Header()
            yield table
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.set_status("Loading...")
            table = self.query_one("#tbl", DataTable)
            for p in store.prototypes:
                table.add_row(p.prototype_id[:8], str(p.strength), p.summary_text)
            self.set_status("")

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
                m.raw_text
                for m in store.memories
                if m.memory_id in proto.constituent_memory_ids
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

    class QueryScreen(StatusMixin):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Ask a question and press Enter", id="hint")
            yield Input(id="query")
            yield TextLog(id="answers")
            yield Static("", id="status")
            yield Footer()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.set_status("Querying...")
            event.input.disabled = True
            res = agent.query(event.value, top_k_prototypes=3, top_k_memories=3)
            log = self.query_one("#answers", TextLog)
            log.clear()
            for p in res.get("prototypes", []):
                log.write_line(f"{p['sim']:.2f} {p['summary']}")
            for m in res.get("memories", []):
                log.write_line(f"  {m['text']}")
            event.input.value = ""
            event.input.disabled = False
            self.set_status("")

    class ChatScreen(StatusMixin):
        BINDINGS = [("escape", "app.pop_screen", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            self.feed = TextLog(id="feed")
            yield self.feed
            self.input = Input(id="msg")
            yield self.input
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.input.focus()
            talk_mgr.register_listener(session_id, "tui", self._on_message)

        def on_unmount(self) -> None:
            talk_mgr.unregister_listener(session_id, "tui")

        def _on_message(self, sender: str, message: str) -> None:
            self.app.call_from_thread(
                self.feed.write_line, f"{sender}: {message}"
            )

        def on_input_submitted(self, event: Input.Submitted) -> None:
            talk_mgr.post_message(session_id, "user", event.value)
            event.input.value = ""

    class ConsoleScreen(StatusMixin):
        BINDINGS = []

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            self.text_log = TextLog(id="console")
            yield self.text_log
            suggestions = [
                "/ingest ",
                "/query ",
                "/beliefs",
                "/stats",
                "/install-models",
                "/talk",
                "/log ",
                "/exit",
                "/quit",
                "/help",
                "/?",
            ]
            self.input = TabAutocompleteInput(
                placeholder="/help for commands", id="cmd", suggestions=suggestions
            )
            yield self.input
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.input.focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            cmd = event.value.strip()
            event.input.value = ""
            if cmd.startswith("/ingest "):
                text = cmd[len("/ingest ") :]
                self.set_status("Ingesting...")
                event.input.disabled = True
                results = agent.add_memory(text)
                for res in results:
                    if res.get("spawned"):
                        msg = f"spawned prototype {res['prototype_id']}"
                    else:
                        msg = f"added to {res['prototype_id']}"
                    self.text_log.write_line(msg)
                self.set_status("Memory ingested.")
                event.input.disabled = False
            elif cmd.startswith("/query "):
                q = cmd[len("/query ") :]
                self.set_status("Querying...")
                event.input.disabled = True
                res = agent.query(q, top_k_prototypes=3, top_k_memories=3)
                for p in res.get("prototypes", []):
                    self.text_log.write_line(f"{p['sim']:.2f} {p['summary']}")
                for m in res.get("memories", []):
                    self.text_log.write_line(f"  {m['text']}")
                self.set_status("")
                event.input.disabled = False
            elif cmd == "/stats":
                usage = _disk_usage(store_path)
                self.text_log.write_line(f"disk: {usage} bytes")
                self.text_log.write_line(f"memories: {len(store.memories)}")
                self.text_log.write_line(f"prototypes: {len(store.prototypes)}")
            elif cmd == "/beliefs":
                self.app.push_screen(BeliefScreen())
            elif cmd == "/talk":
                self.app.push_screen(ChatScreen())
            elif cmd.startswith("/log "):
                path = Path(cmd[len("/log ") :]).expanduser()
                if not path.is_absolute():
                    path = store_path / path
                configure_logging(path)
                self.text_log.write_line(f"logging to {path}")
            elif cmd == "/install-models":
                self.set_status("Installing models...")
                msg = _install_models()
                self.text_log.write_line(msg)
                self.set_status("")
                if msg.startswith("error"):
                    self.app.push_screen(MessageModal(msg))
            elif cmd in ("/exit", "/quit"):
                self.app.push_screen(ExitScreen())
            elif cmd in ("/help", "/?"):
                self.text_log.write_line("/ingest TEXT - add memory")
                self.text_log.write_line("/query TEXT  - search")
                self.text_log.write_line("/beliefs     - list prototypes")
                self.text_log.write_line("/stats       - show stats")
                self.text_log.write_line("/install-models - download models")
                self.text_log.write_line("/talk        - chat session")
                self.text_log.write_line("/log PATH   - write debug log")
                self.text_log.write_line("/exit        - quit")
            elif cmd:
                results = agent.add_memory(cmd)
                for res in results:
                    if res.get("spawned"):
                        msg = f"spawned prototype {res['prototype_id']}"
                    else:
                        msg = f"added to {res['prototype_id']}"
                    self.text_log.write_line(msg)
                try:
                    cues = MemoryCueRenderer().render(
                        [p["summary"] for p in agent.query(cmd, top_k_prototypes=3, top_k_memories=0)["prototypes"]]
                    )
                    parts = [cues] if cues else []
                    for proto in store.prototypes:
                        parts.append(f"{proto.prototype_id}: {proto.summary_text}")
                    for mem in store.memories:
                        parts.append(f"{mem.memory_id}: {mem.raw_text}")
                    context = "\n".join(parts)
                    prompt = f"{context}\nUser: {cmd}\nAssistant:"
                    from .local_llm import LocalChatModel
                    llm = LocalChatModel()
                    prompt = llm.prepare_prompt(agent, prompt)
                    self.set_status("Querying...")
                    reply = llm.reply(prompt)
                    self.text_log.write_line(reply)
                    self.set_status("")
                except Exception as exc:  # pragma: no cover - runtime errors
                    self.text_log.write_line(f"Error: {exc}", style="red")
                    self.set_status("LLM error", error=True)

    class GroupSessionScreen(StatusMixin):
        """IRC-style group chat interface."""

        BINDINGS = [("escape", "leave", "Back")]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            table = DataTable(id="participants")
            table.add_columns("participant")
            feed = TextLog(id="feed", highlight=False)
            yield Header()
            yield Container(table, feed, id="sess")
            suggestions = ["/invite ", "/kick ", "/end"]
            self.input = TabAutocompleteInput(
                placeholder="message", id="msg", suggestions=suggestions
            )
            yield self.input
            yield Static("", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.input.focus()
            talk_mgr.register_listener(session_id, "tui", self._on_msg)
            self._refresh()

        def on_unmount(self) -> None:
            talk_mgr.unregister_listener(session_id, "tui")

        def _refresh(self) -> None:
            table = self.query_one("#participants", DataTable)
            table.clear()
            table.add_row("User")
            sess = talk_mgr.get_session(session_id)
            for pid in sess.agents:
                table.add_row(pid)

        def _on_msg(self, sender: str, text: str) -> None:
            name = "User" if sender == "tui" else Path(sender).name
            log = self.query_one("#feed", TextLog)
            log.write_line(f"[{name}]: {text}")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            msg = event.value.strip()
            event.input.value = ""
            if msg.startswith("/invite "):
                path = msg[len("/invite ") :].strip()
                talk_mgr.invite_brain(session_id, path)
                self._on_msg("system", f"invited {path}")
                self._refresh()
            elif msg.startswith("/kick "):
                bid = msg[len("/kick ") :].strip()
                talk_mgr.kick_brain(session_id, bid)
                self._on_msg("system", f"kicked {bid}")
                self._refresh()
            elif msg == "/end":
                self.action_leave()
            elif msg:
                self._on_msg("tui", msg)
                talk_mgr.post_message(session_id, "tui", msg)

        def action_leave(self) -> None:
            self.app.pop_screen()

    class StatsScreen(StatusMixin):
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
            yield Static("", id="status")
            yield Footer()

    class ExitScreen(StatusMixin):
        BINDINGS = [
            ("y", "yes", "Yes"),
            ("n", "no", "No"),
        ]

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            yield Static("Save brain as zip? [Y/N]", id="exit", markup=False)
            yield Static("", id="status")
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
            ("q", "push_screen('exit')", "Quit"),
            ("f5", "push_screen('stats')", "Stats"),
            ("f6", "push_screen('group')", "Group"),
        ]

        SCREENS = {
            "help": HelpScreen,
            "console": ConsoleScreen,
            "beliefs": BeliefScreen,
            "stats": StatsScreen,
            "group": GroupSessionScreen,
            "talk": ChatScreen,
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
