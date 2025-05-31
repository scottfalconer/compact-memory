from __future__ import annotations

from pathlib import Path

from ..agent import Agent
from ..json_npy_store import JsonNpyVectorStore
from ..config import DEFAULT_BRAIN_PATH
from ..embedding_pipeline import get_embedding_dim
from ..talk_session import TalkSessionManager

from .helpers import _install_models
from .screens import WizardApp, set_context


def run_tui(path: str = DEFAULT_BRAIN_PATH) -> None:
    """Launch the Textual wizard."""
    try:
        import textual  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Textual is required for the TUI") from exc

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
            dim = get_embedding_dim()
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
        store = JsonNpyVectorStore(str(store_path), embedding_dim=dim)

    agent = Agent(store)
    talk_mgr = TalkSessionManager()
    session_id = talk_mgr.create_session([store_path])

    set_context(agent, store, store_path, talk_mgr, session_id)

    WizardApp().run()


__all__ = ["run_tui", "_install_models"]
