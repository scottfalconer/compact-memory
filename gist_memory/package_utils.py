from __future__ import annotations

"""Utilities for working with strategy packages."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

from .compression.strategies_abc import CompressionStrategy


REQUIRED_FIELDS = {
    "package_format_version",
    "strategy_id",
    "strategy_class_name",
    "strategy_module",
    "display_name",
    "version",
    "authors",
    "description",
}


def load_manifest(path: Path) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text())
    return dict(data or {})


def validate_manifest(manifest: Dict[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")
    return errors


def import_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strategy_class(package_dir: Path, manifest: Dict[str, Any]) -> type[CompressionStrategy]:
    module_name = manifest["strategy_module"]
    class_name = manifest["strategy_class_name"]
    module_path = package_dir / f"{module_name}.py"
    module = import_module_from_path(module_name, module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {module_name}")
    if not issubclass(cls, CompressionStrategy):
        raise TypeError(f"{class_name} is not a CompressionStrategy")
    return cls

__all__ = [
    "load_manifest",
    "validate_manifest",
    "import_module_from_path",
    "load_strategy_class",
]
