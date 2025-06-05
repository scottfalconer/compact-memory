from __future__ import annotations

"""Discovery and registration of compression engine plugins."""

from dataclasses import dataclass
from pathlib import Path
import importlib.metadata as metadata
import importlib.util
import logging
import os
import uuid
from typing import Iterable

from platformdirs import user_data_dir

from .engine_registry import register_compression_engine, get_engine_metadata
from .engines import BaseCompressionEngine
from .package_utils import load_manifest

PLUGIN_ENV_VAR = "COMPACT_MEMORY_ENGINES_PATH"
ENTRYPOINT_GROUP = "compact_memory.engines"
DEFAULT_PLUGIN_DIR = (
    Path(user_data_dir("compact_memory", "CompactMemoryTeam")) / "engines"
)

_loaded = False


@dataclass
class PluginPath:
    path: Path
    source_label: str


def _iter_local_plugin_paths() -> Iterable[PluginPath]:
    env = os.getenv(PLUGIN_ENV_VAR)
    if env:
        for raw in env.split(os.pathsep):
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
            if not isinstance(cls, type) or not issubclass(cls, BaseCompressionEngine):
                raise TypeError("Entry point is not a BaseCompressionEngine")
            dist_name = None
            version = None
            try:
                if ep.dist:
                    dist_name = ep.dist.metadata.get("Name")
                    version = ep.dist.version
            except Exception:
                pass
            engine_id = getattr(cls, "id")
            display_name = getattr(cls, "display_name", engine_id)
            prev = get_engine_metadata(engine_id)
            if prev:
                logging.info(
                    "Entry point plugin '%s' overrides %s engine '%s'",
                    ep.value,
                    prev.get("source"),
                    engine_id,
                )
            register_compression_engine(
                engine_id,
                cls,
                display_name=display_name,
                version=version,
                source=f"plugin ({dist_name or 'unknown'})",
            )
        except Exception as exc:
            logging.warning("Failed to load entry point %s: %s", ep.value, exc)


def _load_engine_class_from_module(
    module_file: str, class_name: str
) -> type[BaseCompressionEngine]:
    path = Path(module_file)
    spec = importlib.util.spec_from_file_location(
        f"compact_memory.engine_pkg.{path.stem}_{uuid.uuid4().hex}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {module_file}")
    if not isinstance(cls, type) or not issubclass(cls, BaseCompressionEngine):
        raise TypeError(f"{class_name} is not a BaseCompressionEngine")
    return cls


def _load_local_plugins() -> None:
    for pp in _iter_local_plugin_paths():
        if not pp.path.exists():
            continue
        # First, load any standalone .py files that define engines
        for py_file in sorted(p for p in pp.path.glob("*.py") if p.is_file()):
            if py_file.name == "__init__.py":
                continue
            try:
                mod_spec = importlib.util.spec_from_file_location(
                    f"compact_memory.engine_file.{py_file.stem}_{uuid.uuid4().hex}",
                    py_file,
                )
                if mod_spec is None or mod_spec.loader is None:
                    raise ImportError(f"Cannot load module from {py_file}")
                module = importlib.util.module_from_spec(mod_spec)
                mod_spec.loader.exec_module(module)
                for obj in module.__dict__.values():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BaseCompressionEngine)
                        and obj is not BaseCompressionEngine
                    ):
                        engine_id = getattr(obj, "id")
                        display_name = getattr(obj, "display_name", engine_id)
                        prev = get_engine_metadata(engine_id)
                        if prev:
                            logging.info(
                                "Local plugin '%s' overrides %s engine '%s'",
                                py_file,
                                prev.get("source"),
                                engine_id,
                            )
                        register_compression_engine(
                            engine_id,
                            obj,
                            display_name=display_name,
                            version=None,
                            source=pp.source_label,
                        )
            except Exception as exc:
                logging.warning("Failed loading plugin from %s: %s", py_file, exc)
        # Next, load package-style plugins
        for pkg_dir in sorted(p for p in pp.path.iterdir() if p.is_dir()):
            try:
                manifest = load_manifest(pkg_dir / "engine_package.yaml")
                engine_id = manifest.get("engine_id")
                display_name = manifest.get("display_name", engine_id)
                version = manifest.get("version")
                module_file = pkg_dir / f"{manifest['engine_module']}.py"
                cls = _load_engine_class_from_module(
                    str(module_file), manifest["engine_class_name"]
                )
                prev = get_engine_metadata(engine_id)
                if prev:
                    logging.info(
                        "Local plugin '%s' overrides %s engine '%s'",
                        pkg_dir,
                        prev.get("source"),
                        engine_id,
                    )
                register_compression_engine(
                    engine_id,
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
