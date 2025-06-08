from __future__ import annotations

"""Compression engine utilities and dataclasses."""

import os
import importlib  # Keep for load_engine
from pathlib import Path  # Keep for load_engine

# Import from the new base.py
from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace

# Import registry functions
from .registry import get_compression_engine
from compact_memory.plugin_loader import load_plugins


# Load plugins (including any built-in engines registered via entry points)
load_plugins()


def load_engine(path: str | os.PathLike) -> BaseCompressionEngine:
    """Load a compression engine from ``path`` using its manifest."""
    p = Path(path)
    with open(p / "engine_manifest.json", "r", encoding="utf-8") as fh:
        import json  # Local import as it's only used here

        manifest = json.load(fh)
    engine_id = manifest.get("engine_id")
    engine_class_path = manifest.get("engine_class")
    engine_config = manifest.get("config", {})
    cls = None
    if engine_class_path:
        mod_name, cls_name = engine_class_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    elif engine_id:
        # ``get_compression_engine`` will locate engines that have been
        # registered via the plugin system.
        cls = get_compression_engine(engine_id)
    else:
        raise ValueError("Engine manifest missing engine_id or engine_class")

    if not issubclass(cls, BaseCompressionEngine):  # type: ignore # cls could be None then caught by next line, but logic implies it's found
        raise TypeError(
            f"Loaded class {engine_class_path or engine_id} is not a BaseCompressionEngine."
        )

    engine = cls(config=engine_config)
    engine.load(p)
    return engine


__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "load_engine",
    # Specific engine classes are not exported here. They can be imported
    # from their defining packages or retrieved via ``get_compression_engine``.
]
