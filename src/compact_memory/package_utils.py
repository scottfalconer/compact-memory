from __future__ import annotations

"""Utilities for working with engine packages."""

import importlib.util
import importlib
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from compact_memory.engines.base import BaseCompressionEngine

# Import directly from the ``base`` module to avoid importing ``engines``,
# which triggers plugin loading and can cause circular imports during start-up.

# ``BaseCompressionEngine`` is imported only within the helper functions to
# avoid triggering plugin loading during module import. This prevents circular
# dependency issues when ``compact_memory.plugin_loader`` imports this module.


REQUIRED_FIELDS = {
    "package_format_version",
    "engine_id",
    "engine_class_name",
    "engine_module",
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


def load_engine_class_from_module(
    module_file_path: str, class_name: str
) -> Type["BaseCompressionEngine"]:
    """Load and return ``class_name`` from ``module_file_path``."""

    module_path = Path(module_file_path)
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_file_path}")

    unique_name = f"compact_memory.packages.{module_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_file_path}")
    module = importlib.util.module_from_spec(spec)

    sys_path = list(sys.path)
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = sys_path

    from compact_memory.engines.base import BaseCompressionEngine as CompressionEngine

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {module_file_path}")
    if not isinstance(cls, type) or not issubclass(cls, CompressionEngine):
        raise TypeError(f"{class_name} is not a CompressionEngine")
    return cls


def load_engine_class(
    package_dir: Path, manifest: Dict[str, Any]
) -> type["BaseCompressionEngine"]:
    module_name = manifest["engine_module"]
    class_name = manifest["engine_class_name"]
    module_path = package_dir / f"{module_name}.py"
    module = import_module_from_path(module_name, module_path)
    from compact_memory.engines.base import BaseCompressionEngine as CompressionEngine

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {module_name}")
    if not issubclass(cls, CompressionEngine):
        raise TypeError(f"{class_name} is not a CompressionEngine")
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
    """Validate an engine package directory.

    Returns a tuple of (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = package_dir / "engine_package.yaml"
    if not manifest_path.exists():
        errors.append("engine_package.yaml not found")
        return errors, warnings

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        errors.append(f"Invalid engine_package.yaml: {exc}")
        return errors, warnings

    errors.extend(validate_manifest(manifest))

    module_name = manifest.get("engine_module")
    if module_name:
        module_path = package_dir / f"{module_name}.py"
        if not module_path.exists():
            errors.append(f"{module_path.name} not found")
        else:
            try:
                load_engine_class(package_dir, manifest)
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
    "load_engine_class_from_module",
    "load_engine_class",
    "check_requirements_installed",
    "validate_package_dir",
]
