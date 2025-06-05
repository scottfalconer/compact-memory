from __future__ import annotations

"""Discovery and registration of compression engine plugins."""

from dataclasses import dataclass
from pathlib import Path
import importlib.metadata as metadata
import logging
import os
from typing import Iterable

from platformdirs import user_data_dir

from CompressionEngine.core.registry import register_compression_engine, get_engine_metadata # Updated path and function names
from CompressionEngine.core.engines_abc import CompressionEngine # Updated path and class name
from .package_utils import (
    validate_package_dir, # Assuming this function will be internally updated to look for engine files
    load_manifest,
    load_engine_class_from_module, # Updated function name
)


PLUGIN_ENV_VAR = "COMPACT_MEMORY_PLUGINS_PATH" # This might need to change if it's too strategy-specific
ENTRYPOINT_GROUP = "compact_memory.engines" # CRITICAL: Updated entry point group
DEFAULT_PLUGIN_DIR = Path(user_data_dir("compact_memory", "CompactMemoryTeam")) / "plugins"


_loaded = False


@dataclass
class PluginPath:
    path: Path
    source_label: str


def _iter_local_plugin_paths() -> Iterable[PluginPath]:
    env = os.getenv(PLUGIN_ENV_VAR)
    if env:
        for idx, raw in enumerate(env.split(os.pathsep)):
            if not raw:
                continue
            yield PluginPath(Path(raw), f"local ({Path(raw).name})")
    yield PluginPath(DEFAULT_PLUGIN_DIR, f"local ({DEFAULT_PLUGIN_DIR.name})")


def load_plugins() -> None:
    """Discover and register all compression engine plugins."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    _load_entrypoint_plugins()
    _load_local_plugins()


def _load_entrypoint_plugins() -> None:
    try:
        eps = metadata.entry_points(group=ENTRYPOINT_GROUP)
    except TypeError:  # pragma: no cover - older Python
        eps = metadata.entry_points().get(ENTRYPOINT_GROUP, [])

    for ep in eps:
        try:
            cls = ep.load()
            if not isinstance(cls, type) or not issubclass(cls, CompressionEngine): # Updated class name
                raise TypeError("Entry point is not a CompressionEngine") # Updated error message
            dist_name = None
            version = None
            try:
                if ep.dist:
                    dist_name = ep.dist.metadata.get("Name")
                    version = ep.dist.version
            except Exception:
                pass
            engine_id = getattr(cls, "id") # Updated variable name
            display_name = getattr(cls, "display_name", engine_id)
            prev = get_engine_metadata(engine_id) # Updated function call and variable
            if prev:
                logging.info(
                    "Entry point plugin '%s' overrides %s engine '%s'", # Updated text
                    ep.value,
                    prev.get("source"),
                    engine_id, # Updated variable
                )
            register_compression_engine( # Updated function name
                engine_id, # Updated variable
                cls,
                display_name=display_name,
                version=version,
                source=f"plugin ({dist_name or 'unknown'})",
            )
        except Exception as exc:
            logging.warning("Failed to load entry point %s: %s", ep.value, exc)


def _load_local_plugins() -> None:
    for pp in _iter_local_plugin_paths():
        if not pp.path.exists():
            continue
        for pkg_dir in sorted(p for p in pp.path.iterdir() if p.is_dir()):
            try:
                errors, _ = validate_package_dir(pkg_dir) # validate_package_dir needs to check for engine_package.yaml etc.
                if errors:
                    logging.warning(
                        "Invalid plugin at %s: %s", pkg_dir, ", ".join(errors)
                    )
                    continue
                manifest = load_manifest(pkg_dir / "engine_package.yaml") # Updated filename
                engine_id = manifest.get("engine_id") # Updated key
                display_name = manifest.get("display_name", engine_id)
                version = manifest.get("version")
                module_file = pkg_dir / f"{manifest['engine_module']}.py" # Updated key
                cls = load_engine_class_from_module( # Updated function name
                    str(module_file), manifest["engine_class_name"] # Updated key
                )
                prev = get_engine_metadata(engine_id) # Updated function call and variable
                if prev:
                    logging.info(
                        "Local plugin '%s' overrides %s engine '%s'", # Updated text
                        pkg_dir,
                        prev.get("source"),
                        engine_id, # Updated variable
                    )
                register_compression_engine( # Updated function name
                    engine_id, # Updated variable
                    cls,
                    display_name=display_name,
                    version=version,
                    source=pp.source_label,
                )
            except Exception as exc:
                logging.warning("Failed loading plugin from %s: %s", pkg_dir, exc)


__all__ = [
    "load_plugins",
    "PLUGIN_ENV_VAR",
    "ENTRYPOINT_GROUP", # This now refers to "compact_memory.engines"
    "DEFAULT_PLUGIN_DIR",
]
