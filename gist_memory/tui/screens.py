from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input, DataTable

from ..autocomplete_input import TabAutocompleteInput
from ..memory_cues import MemoryCueRenderer
from ..logging_utils import configure_logging
from ..agent import Agent
from ..json_npy_store import JsonNpyVectorStore
from ..talk_session import TalkSessionManager
from ..utils import format_ingest_results

from .helpers import (
    _install_models,
    _path_suggestions,
    _brain_path_suggestions,
)

try:  # Textual 0.x
    from textual.widgets import TextLog  # type: ignore
except Exception:  # pragma: no cover - Textual >=1.0 renamed the widget
    from textual.widgets import Log as TextLog  # type: ignore

# Context variables set by run_tui
agent: Agent
store: JsonNpyVectorStore
store_path: Path
talk_mgr: TalkSessionManager
session_id: str


def set_context(
    a: Agent, s: JsonNpyVectorStore, path: Path, mgr: TalkSessionManager, sid: str
) -> None:
    global agent, store, store_path, talk_mgr, session_id
    agent = a
    store = s
    store_path = path
    talk_mgr = mgr
    session_id = sid


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
            "/params           - agent parameters\n"
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
            "Use F5 for stats, F7 for conflicts and Q to quit."
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
                self.set_status("error reading file", error=True)
                event.input.value = ""
                event.input.disabled = False
                return
        else:
            text = event.value

        results = agent.add_memory(text)
        log = self.query_one("#log", TextLog)
        for line in format_ingest_results(agent, results):
            log.write_line(line)
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
        protos = agent.get_prototypes_view()
        for p in protos:
            table.add_row(p["id"][:8], str(p["strength"]), p["summary"])
        self.set_status("")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        idx = event.row_key
        if idx is None:
            return
        proto = store.prototypes[int(idx)]
        self.app.context["selected_proto"] = proto.prototype_id

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.row_key
        proto = store.prototypes[int(idx)]
        mems = [
            m.raw_text
            for m in store.memories
            if m.memory_id in proto.constituent_memory_ids
        ][:3]
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
            log.write_line(f"{p['id']} {p['summary']} ({p['sim']:.2f})")
        for m in res.get("memories", []):
            log.write_line(f"  {m['text']} ({m['sim']:.2f})")
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
        self.app.call_from_thread(self.feed.write_line, f"{sender}: {message}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        talk_mgr.post_message(session_id, "user", event.value)
        event.input.value = ""


class ConsoleScreen(StatusMixin):
    BINDINGS = []

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header()
        self.text_log = TextLog(id="console")
        yield self.text_log
        base_suggestions = [
            "/ingest ",
            "/query ",
            "/beliefs",
            "/stats",
            "/install-models",
            "/talk",
            "/params",
            "/log ",
            "/exit",
            "/quit",
            "/help",
            "/?",
        ]

        self.recent_queries: list[str] = []

        def suggest(value: str) -> list[str]:
            if value.startswith("/log "):
                prefix = value[len("/log ") :]
                return ["/log " + p for p in _path_suggestions(prefix)]
            if value.startswith("/query "):
                prefix = value[len("/query ") :]
                opts = [q for q in self.recent_queries if q.startswith(prefix)]
                for proto in store.prototypes:
                    s = proto.summary_text
                    if s.startswith(prefix):
                        opts.append(s)
                return ["/query " + o for o in opts][:10]
            return [s for s in base_suggestions if s.startswith(value)]

        self.input = TabAutocompleteInput(
            placeholder="/help for commands", id="cmd", suggestions=suggest
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
            for line in format_ingest_results(agent, results):
                self.text_log.write_line(line)
            self.set_status("Memory ingested.")
            event.input.disabled = False
        elif cmd.startswith("/query "):
            q = cmd[len("/query ") :]
            self.recent_queries.insert(0, q)
            self.recent_queries = self.recent_queries[:10]
            self.set_status("Querying...")
            event.input.disabled = True
            res = agent.query(q, top_k_prototypes=3, top_k_memories=3)
            for p in res.get("prototypes", []):
                self.text_log.write_line(f"{p['id']} {p['summary']} ({p['sim']:.2f})")
            for m in res.get("memories", []):
                self.text_log.write_line(f"  {m['text']} ({m['sim']:.2f})")
            self.set_status("")
            event.input.disabled = False
        elif cmd == "/stats":
            stats = agent.get_statistics()
            self.text_log.write_line(f"disk: {stats.get('disk_usage', 0)} bytes")
            self.text_log.write_line(f"memories: {stats.get('memories', 0)}")
            self.text_log.write_line(f"prototypes: {stats.get('prototypes', 0)}")
        elif cmd == "/beliefs":
            self.app.push_screen(BeliefScreen())
        elif cmd == "/talk":
            self.app.push_screen(ChatScreen())
        elif cmd == "/params":
            self.app.push_screen(ParamsScreen())
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
            self.text_log.write_line("/params      - agent parameters")
            self.text_log.write_line("/log PATH   - write debug log")
            self.text_log.write_line("/exit        - quit")
        elif cmd:
            results = agent.add_memory(cmd)
            for line in format_ingest_results(agent, results):
                self.text_log.write_line(line)
            try:
                cues = MemoryCueRenderer().render(
                    [
                        p["summary"]
                        for p in agent.query(cmd, top_k_prototypes=3, top_k_memories=0)[
                            "prototypes"
                        ]
                    ]
                )
                parts = [cues] if cues else []
                for proto in store.prototypes:
                    parts.append(f"{proto.prototype_id}: {proto.summary_text}")
                for mem in store.memories:
                    parts.append(f"{mem.memory_id}: {mem.raw_text}")
                context = "\n".join(parts)
                prompt = f"{context}\nUser: {cmd}\nAssistant:"
                from ..local_llm import LocalChatModel

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
        base_suggestions = ["/invite ", "/kick ", "/end"]

        def suggest(value: str) -> list[str]:
            if value.startswith("/invite "):
                prefix = value[len("/invite ") :]
                return [
                    "/invite " + p
                    for p in _brain_path_suggestions(store_path.parent, prefix)
                ]
            return [s for s in base_suggestions if s.startswith(value)]

        self.input = TabAutocompleteInput(
            placeholder="message", id="msg", suggestions=suggest
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
        stats = agent.get_statistics()
        table.add_row("disk", f"{stats.get('disk_usage', 0)} bytes")
        table.add_row("memories", str(stats.get("memories", 0)))
        table.add_row("prototypes", str(stats.get("prototypes", 0)))
        table.add_row("tau", str(stats.get("tau", 0)))
        table.add_row("updated", stats.get("updated", ""))
        yield Header()
        yield table
        yield Static("", id="status")
        yield Footer()


class ConflictListScreen(StatusMixin):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:  # type: ignore[override]
        table = DataTable(id="conflicts")
        table.add_columns("prototype", "memory A", "memory B", "reason")
        yield Header()
        yield table
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.set_status("Loading...")
        path = store_path / "conflicts.jsonl"
        table = self.query_one("#conflicts", DataTable)
        if not path.exists():
            self.set_status("No conflicts logged")
            return
        try:
            lines = path.read_text().splitlines()
        except Exception as exc:  # pragma: no cover - runtime errors
            self.set_status(f"error reading log: {exc}", error=True)
            return

        mem_map = {m.memory_id: m.raw_text for m in store.memories}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("prototype_id", "")
            text_a = rec.get("text_a")
            text_b = rec.get("text_b")
            if text_a is None:
                mid_a = rec.get("memory_a") or rec.get("memory_id_a")
                text_a = mem_map.get(mid_a, mid_a or "")
            if text_b is None:
                mid_b = rec.get("memory_b") or rec.get("memory_id_b")
                text_b = mem_map.get(mid_b, mid_b or "")
            reason = rec.get("reason", "")
            table.add_row(pid[:8], text_a, text_b, reason)
        self.set_status("")


class ParamsScreen(StatusMixin):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:  # type: ignore[override]
        table = DataTable(id="params")
        table.add_columns("key", "value")
        yield Header()
        yield table
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#params", DataTable)
        table.add_row("tau", str(agent.similarity_threshold))
        chunk_cfg = getattr(agent.chunker, "config", lambda: {})()
        chunk_id = chunk_cfg.pop("id", agent.chunker.__class__.__name__)
        table.add_row("chunker", str(chunk_id))
        for k, v in chunk_cfg.items():
            table.add_row(f"chunker.{k}", str(v))

        creator = agent.summary_creator
        table.add_row("memory_creator", creator.__class__.__name__)
        for k, v in vars(creator).items():
            table.add_row(f"memory_creator.{k}", str(v))


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
        ("f7", "push_screen('conflicts')", "Conflicts"),
    ]

    SCREENS = {
        "help": HelpScreen,
        "console": ConsoleScreen,
        "beliefs": BeliefScreen,
        "stats": StatsScreen,
        "params": ParamsScreen,
        "group": GroupSessionScreen,
        "conflicts": ConflictListScreen,
        "talk": ChatScreen,
        "exit": ExitScreen,
    }

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

    def on_screen_resume(self, event: Screen.Resume) -> None:  # pragma: no cover
        if isinstance(event.screen, ExitScreen):
            return
        store.save()


__all__ = [
    "set_context",
    "WizardApp",
    "HelpScreen",
    "WelcomeScreen",
    "IngestScreen",
    "BeliefScreen",
    "DetailScreen",
    "QueryScreen",
    "ChatScreen",
    "ConsoleScreen",
    "GroupSessionScreen",
    "ParamsScreen",
    "StatsScreen",
    "ConflictListScreen",
    "ExitScreen",
]
