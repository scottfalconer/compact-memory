from __future__ import annotations

"""Compression engine utilities and dataclasses."""

import os
from typing import Any, Dict # Removed List, Optional, Sequence, Callable as they are used in base.py
# Removed dataclasses, json, uuid, np, faiss as they are used in base.py or other submodules

# Import from the new base.py
from .base import BaseCompressionEngine, CompressedMemory, CompressionTrace

# Import PrototypeEngine from one level up
# from ..prototype_engine import PrototypeEngine # REMOVED TO BREAK CYCLE

# Imports needed for registration and load_engine
import importlib
from pathlib import Path
from .registry import get_compression_engine, register_compression_engine
from .no_compression_engine import NoCompressionEngine # NoCompressionEngine is self-contained enough
from .first_last_engine import FirstLastEngine # Import FirstLastEngine

# Register NoCompressionEngine here as it doesn't create cycles.
# PrototypeEngine will be registered in cli.py
# Register FirstLastEngine here as well.
register_compression_engine(NoCompressionEngine.id, NoCompressionEngine, display_name="No Compression", source="built-in")
register_compression_engine(FirstLastEngine.id, FirstLastEngine, display_name="First/Last Chunks", source="built-in") # Changed source to built-in from contrib for consistency


# Keep load_engine here as it uses get_compression_engine which might be in .registry
# and also deals with dynamic imports based on manifest.
# Or, if get_compression_engine is also moved to base or a utils, load_engine could move too.
# For now, keeping it here and adjusting its imports if necessary.

def load_engine(path: str | os.PathLike) -> BaseCompressionEngine:
    """Load a compression engine from ``path`` using its manifest."""

    # imports already moved to top level: importlib, Path, get_compression_engine
    p = Path(path)
    with open(p / "engine_manifest.json", "r", encoding="utf-8") as fh:
        import json # Moved json import here as it's only used in load_engine
        manifest = json.load(fh)
    engine_id = manifest.get("engine_id")
    engine_class_path = manifest.get("engine_class") # Renamed for clarity
    engine_config = manifest.get("config", {})
    cls = None
    if engine_class_path:
        mod_name, cls_name = engine_class_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    elif engine_id:
        cls = get_compression_engine(engine_id) # This needs to be resolvable
    else:
        raise ValueError("Engine manifest missing engine_id or engine_class")

    # Instantiate the engine, passing the config if available
    # Ensure the class (cls) is a subclass of BaseCompressionEngine before instantiation
    if not issubclass(cls, BaseCompressionEngine):
        raise TypeError(f"Loaded class {cls_name or engine_id} is not a BaseCompressionEngine.")

    engine = cls(config=engine_config) # Pass engine_config as the config argument

    engine.load(p) # The engine's own load method
    return engine


__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "load_engine",
    # "PrototypeEngine", # REMOVED FROM HERE
]
