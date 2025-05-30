from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
import time

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .embedding_pipeline import embed_text, EmbeddingDimensionMismatchError
from .chunker import SentenceWindowChunker, Chunker


def _load_agent(path: Path) -> Agent:
    """Load a persisted agent from ``path``.

    This mirrors the logic of the CLI helper for convenience.
    """
    try:
        store = JsonNpyVectorStore(path=str(path))
    except Exception as exc:
        raise RuntimeError(f"Error loading agent at {path}: {exc}") from exc
    except EmbeddingDimensionMismatchError:
        dim = int(embed_text(["dim"]).shape[1])
        store = JsonNpyVectorStore(path=str(path), embedding_dim=dim)
    chunker_id = store.meta.get("chunker", "sentence_window")
    chunker_cls: type[Chunker] = SentenceWindowChunker
    if chunker_id == "sentence_window":
        chunker_cls = SentenceWindowChunker
    return Agent(store, chunker=chunker_cls(), similarity_threshold=float(store.meta.get("tau", 0.8)))


@dataclass
class TalkSession:
    session_id: str
    agents: Dict[str, Agent] = field(default_factory=dict)
    listeners: Dict[str, Callable[[str, str], None]] = field(default_factory=dict)
    log: List[Tuple[str, str, float]] = field(default_factory=list)


class TalkSessionManager:
    """Manage multi-agent talk sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, TalkSession] = {}

    # ------------------------------------------------------------------
    def create_session(self, agent_paths: Iterable[str | Path]) -> str:
        """Create a session with agents loaded from ``agent_paths``.

        Returns the new session ID.
        """
        sid = uuid.uuid4().hex
        agents: Dict[str, Agent] = {}
        for p in agent_paths:
            path = Path(p)
            agents[str(path)] = _load_agent(path)
        self._sessions[sid] = TalkSession(session_id=sid, agents=agents)
        return sid

    def end_session(self, session_id: str) -> None:
        """Terminate ``session_id`` if it exists."""
        self._sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> TalkSession:
        return self._sessions[session_id]

    def register_listener(
        self, session_id: str, listener_id: str, callback: Callable[[str, str], None]
    ) -> None:
        """Register ``callback`` to receive messages for ``session_id``."""
        session = self._sessions[session_id]
        session.listeners[listener_id] = callback

    def unregister_listener(self, session_id: str, listener_id: str) -> None:
        """Remove ``listener_id`` from ``session_id`` if present."""
        session = self._sessions[session_id]
        session.listeners.pop(listener_id, None)

    def post_message(self, session_id: str, sender: str, message: str) -> None:
        """Append ``message`` from ``sender`` to the session log and broadcast."""
        session = self._sessions[session_id]
        ts = time.time()
        session.log.append((sender, message, ts))

        for aid, agent in session.agents.items():
            if aid == sender:
                continue
            if hasattr(agent, "receive_channel_message"):
                try:
                    agent.receive_channel_message(sender, message)
                except TypeError:
                    # fallback for old signature
                    agent.receive_channel_message(message)  # type: ignore[arg-type]
            else:
                agent.add_memory(message)

        for lid, cb in session.listeners.items():
            if lid == sender:
                continue
            cb(sender, message)

    # ------------------------------------------------------------------
    def invite_brain(self, session_id: str, brain_path_or_id: str | Path) -> None:
        """Add a brain to an existing session.

        Late-joining brains ingest the full message history so far.
        """
        session = self._sessions[session_id]
        path = Path(brain_path_or_id)
        key = str(path)
        if key in session.agents:
            return
        agent = _load_agent(path)
        for _sender, msg, _ts in session.log:
            agent.add_memory(msg)
        session.agents[key] = agent

    def kick_brain(self, session_id: str, brain_id: str) -> None:
        """Remove ``brain_id`` from the session if present."""
        session = self._sessions[session_id]
        session.agents.pop(brain_id, None)
