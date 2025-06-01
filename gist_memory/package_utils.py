from __future__ import annotations

"""Utilities for working with strategy packages."""

import importlib.util
import importlib
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


def check_requirements_installed(req_file: Path) -> list[str]:
    """Return a list of requirement names that are not currently importable."""
    missing: list[str] = []
    if not req_file.exists():
        return missing
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("==")[0].split(">=")[0].split("<=")[0]
        pkg = pkg.split("[")[0].strip()
        module_name = pkg.replace("-", "_")
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pkg)
    return missing


def validate_package_dir(package_dir: Path) -> tuple[list[str], list[str]]:
    """Validate a strategy package directory.

    Returns a tuple of (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = package_dir / "strategy_package.yaml"
    if not manifest_path.exists():
        errors.append("strategy_package.yaml not found")
        return errors, warnings

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        errors.append(f"Invalid strategy_package.yaml: {exc}")
        return errors, warnings

    errors.extend(validate_manifest(manifest))

    module_name = manifest.get("strategy_module")
    if module_name:
        module_path = package_dir / f"{module_name}.py"
        if not module_path.exists():
            errors.append(f"{module_path.name} not found")
        else:
            try:
                load_strategy_class(package_dir, manifest)
            except Exception as exc:
                errors.append(str(exc))

    if not (package_dir / "requirements.txt").exists():
        warnings.append("requirements.txt not found")
    if not (package_dir / "README.md").exists():
        warnings.append("README.md not found")

    return errors, warnings

__all__ = [
    "load_manifest",
    "validate_manifest",
    "import_module_from_path",
    "load_strategy_class",
    "check_requirements_installed",
    "validate_package_dir",
]
