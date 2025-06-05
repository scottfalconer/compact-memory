from __future__ import annotations

"""Discovery and registration of compression strategy plugins."""

from dataclasses import dataclass
from pathlib import Path
import importlib.metadata as metadata
import logging
import os
from typing import Iterable

from platformdirs import user_data_dir

from CompressionStrategy.core import register_compression_strategy, get_strategy_metadata
from CompressionStrategy.core.strategies_abc import CompressionStrategy
from .package_utils import (
    validate_package_dir,
    load_manifest,
    load_strategy_class_from_module,
)


PLUGIN_ENV_VAR = "COMPACT_MEMORY_PLUGINS_PATH"
ENTRYPOINT_GROUP = "compact_memory.strategies"
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
    """Discover and register all compression strategy plugins."""
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
            if not isinstance(cls, type) or not issubclass(cls, CompressionStrategy):
                raise TypeError("Entry point is not a CompressionStrategy")
            dist_name = None
            version = None
            try:
                if ep.dist:
                    dist_name = ep.dist.metadata.get("Name")
                    version = ep.dist.version
            except Exception:
                pass
            strategy_id = getattr(cls, "id")
            display_name = getattr(cls, "display_name", strategy_id)
            prev = get_strategy_metadata(strategy_id)
            if prev:
                logging.info(
                    "Entry point plugin '%s' overrides %s strategy '%s'",
                    ep.value,
                    prev.get("source"),
                    strategy_id,
                )
            register_compression_strategy(
                strategy_id,
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
                errors, _ = validate_package_dir(pkg_dir)
                if errors:
                    logging.warning(
                        "Invalid plugin at %s: %s", pkg_dir, ", ".join(errors)
                    )
                    continue
                manifest = load_manifest(pkg_dir / "strategy_package.yaml")
                strategy_id = manifest.get("strategy_id")
                display_name = manifest.get("display_name", strategy_id)
                version = manifest.get("version")
                module_file = pkg_dir / f"{manifest['strategy_module']}.py"
                cls = load_strategy_class_from_module(
                    str(module_file), manifest["strategy_class_name"]
                )
                prev = get_strategy_metadata(strategy_id)
                if prev:
                    logging.info(
                        "Local plugin '%s' overrides %s strategy '%s'",
                        pkg_dir,
                        prev.get("source"),
                        strategy_id,
                    )
                register_compression_strategy(
                    strategy_id,
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
    "ENTRYPOINT_GROUP",
    "DEFAULT_PLUGIN_DIR",
]
