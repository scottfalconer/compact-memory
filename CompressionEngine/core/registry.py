from __future__ import annotations

"""Registration utilities for compression engines."""

import warnings
from typing import Dict, Optional as Opt, Type, List, Any

from .engines_abc import CompressionEngine

# Global registry for compression engine classes
_COMPRESSION_ENGINE_REGISTRY: Dict[str, Type[CompressionEngine]] = {}
# Global registry for metadata associated with each compression engine
_COMPRESSION_ENGINE_INFO: Dict[str, Dict[str, Opt[str]]] = {}
# Global registry for configuration classes associated with engines (optional)
_COMPRESSION_ENGINE_CONFIG_REGISTRY: Dict[str, Type[Any]] = {}


def register_compression_engine(
    id: str,
    cls: Type[CompressionEngine],
    *,
    config_cls: Opt[Type[Any]] = None,
    display_name: str | None = None,
    version: str | None = None,
    source: str = "built-in",
) -> None:
    """
    Register a CompressionEngine class ``cls`` under a unique string ``id``.

    Args:
        id: The unique identifier for the compression engine.
        cls: The CompressionEngine class to register.
        config_cls: Optional configuration class associated with this engine.
        display_name: A user-friendly name for the engine. If None, uses ``id``.
        version: Version string for the engine. Defaults to "N/A".
        source: Origin of the engine (e.g., "built-in", "plugin <name>").
    """
    if not issubclass(cls, CompressionEngine):
        raise TypeError(f"Class {cls.__name__} must inherit from CompressionEngine.")
    if id in _COMPRESSION_ENGINE_REGISTRY:
        warnings.warn(
            f"CompressionEngine with id '{id}' already registered. Overwriting.",
            UserWarning,
        )

    _COMPRESSION_ENGINE_REGISTRY[id] = cls
    _COMPRESSION_ENGINE_INFO[id] = {
        "display_name": display_name or id,
        "version": version or "N/A",
        "source": source,
        "class_name": cls.__name__,
        "module_name": cls.__module__,
    }
    if config_cls:
        _COMPRESSION_ENGINE_CONFIG_REGISTRY[id] = config_cls


def get_compression_engine(id: str) -> Type[CompressionEngine]:
    """
    Return the CompressionEngine class registered under ``id``.

    Args:
        id: The unique identifier for the desired compression engine.

    Returns:
        The CompressionEngine class.

    Raises:
        KeyError: If no engine is registered with the given ``id``.
    """
    if id not in _COMPRESSION_ENGINE_REGISTRY:
        raise KeyError(f"Unknown compression engine: {id}")
    return _COMPRESSION_ENGINE_REGISTRY[id]


def available_engines() -> List[str]:
    """Return a sorted list of IDs for all registered compression engines."""
    return sorted(_COMPRESSION_ENGINE_REGISTRY.keys())


def get_engine_metadata(id: str) -> Opt[Dict[str, Opt[str]]]:
    """
    Return metadata for the compression engine registered under ``id``.

    Args:
        id: The unique identifier for the compression engine.

    Returns:
        A dictionary of metadata, or None if the engine is not found.
    """
    return _COMPRESSION_ENGINE_INFO.get(id)


def all_engine_metadata() -> Dict[str, Dict[str, Opt[str]]]:
    """Return a dictionary of all registered compression engines and their metadata."""
    return dict(_COMPRESSION_ENGINE_INFO)


def get_engine_config_class(id: str) -> Opt[Type[Any]]:
    """
    Return the configuration class associated with the compression engine ``id``.

    Args:
        id: The unique identifier for the compression engine.

    Returns:
        The configuration class, or None if not found or not applicable.
    """
    return _COMPRESSION_ENGINE_CONFIG_REGISTRY.get(id)


def validate_engine_id(engine_id: str, expected_type: Opt[type] = None) -> bool:
    """
    Validate if an engine ID is registered and optionally matches an expected type.

    Args:
        engine_id: The ID of the engine to validate.
        expected_type: If provided, checks if the registered engine is a subclass
                       of this type (e.g., a specific CompressionEngine subclass).

    Returns:
        True if the engine ID is valid and matches the type (if specified),
        False otherwise.
    """
    if engine_id not in _COMPRESSION_ENGINE_REGISTRY:
        warnings.warn(f"Engine ID '{engine_id}' not found in registry.", UserWarning)
        return False
    if expected_type:
        engine_class = _COMPRESSION_ENGINE_REGISTRY[engine_id]
        if not issubclass(engine_class, expected_type):
            warnings.warn(
                f"Engine ID '{engine_id}' (class: {engine_class.__name__}) "
                f"does not match expected type {expected_type.__name__}.",
                UserWarning,
            )
            return False
    return True


__all__ = [
    "register_compression_engine",
    "get_compression_engine",
    "available_engines",
    "get_engine_metadata",
    "all_engine_metadata",
    "get_engine_config_class",
    "validate_engine_id",
]
