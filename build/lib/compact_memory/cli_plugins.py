from __future__ import annotations

"""Loading of Typer command groups provided by plugins."""

import logging
import importlib.metadata as metadata
from typing import Iterable

import typer

CLI_ENTRYPOINT_GROUP = "compact_memory.cli"

_loaded = False


def load_cli_plugins(app: typer.Typer) -> None:
    """Load CLI plugins and attach them to ``app``.

    Plugins are discovered via the ``compact_memory.cli`` entry point group. Each
    entry point should load and return either a :class:`typer.Typer` instance or
    a callable that accepts the main application and registers commands.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    try:
        eps = metadata.entry_points(group=CLI_ENTRYPOINT_GROUP)
    except TypeError:  # pragma: no cover - older Python
        eps = metadata.entry_points().get(CLI_ENTRYPOINT_GROUP, [])

    for ep in eps:
        try:
            plugin = ep.load()
            if isinstance(plugin, typer.Typer):
                app.add_typer(plugin, name=ep.name)
            elif callable(plugin):
                plugin(app)
            else:
                logging.warning("CLI plugin %s has unsupported type", ep.value)
        except Exception as exc:
            logging.warning("Failed to load CLI plugin %s: %s", ep.value, exc)


__all__ = ["load_cli_plugins", "CLI_ENTRYPOINT_GROUP"]
