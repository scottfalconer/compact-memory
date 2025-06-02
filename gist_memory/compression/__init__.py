"""Compression strategy interfaces and implementations."""

from __future__ import annotations

from importlib import util
from pathlib import Path

from .strategies_abc import CompressedMemory, CompressionStrategy, CompressionTrace
from .config import StrategyConfig

# ---------------------------------------------------------------------------
# Support legacy simple compression strategies defined in ``compression.py``.
# ``gist_memory.compression`` used to be a single module providing these basic
# utilities.  A package now occupies the name, so we load the legacy module
# explicitly by path to maintain backwards compatibility with imports such as::
#
#     from gist_memory.compression import NoCompression
#
_legacy_path = Path(__file__).resolve().parent.parent / "compression.py"
_spec = util.spec_from_file_location("gist_memory._compression_legacy", _legacy_path)
_legacy = util.module_from_spec(_spec)
assert _spec.loader is not None  # for mypy/static checkers
_spec.loader.exec_module(_legacy)

NoCompression = _legacy.NoCompression
ImportanceCompression = _legacy.ImportanceCompression
register_compression_strategy = _legacy.register_compression_strategy
get_compression_strategy = _legacy.get_compression_strategy
available_strategies = _legacy.available_strategies

__all__ = [
    "CompressedMemory",
    "CompressionStrategy",
    "CompressionTrace",
    "NoCompression",
    "ImportanceCompression",
    "register_compression_strategy",
    "get_compression_strategy",
    "available_strategies",
    "StrategyConfig",
]
