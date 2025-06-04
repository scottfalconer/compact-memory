from pathlib import Path

import pytest

from compact_memory.compression.registry import (
    _COMPRESSION_REGISTRY,
    _COMPRESSION_INFO,
    get_strategy_metadata,
)
from compact_memory.plugin_loader import load_plugins, PLUGIN_ENV_VAR


def _create_plugin(pkg_dir: Path) -> None:
    pkg_dir.mkdir()
    (pkg_dir / "strategy.py").write_text(
        """
from compact_memory.compression.strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)

class DummyPluginStrategy(CompressionStrategy):
    id = 'dummy_plugin'

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        return CompressedMemory(text='x'), CompressionTrace(
            strategy_name=self.id,
            strategy_params={},
            input_summary={},
            steps=[],
        )
"""
    )
    (pkg_dir / "strategy_package.yaml").write_text(
        """
package_format_version: '1.0'
strategy_id: dummy_plugin
strategy_class_name: DummyPluginStrategy
strategy_module: strategy
display_name: Dummy Plugin
version: '0.1'
authors: ['Tester']
description: Test plugin
"""
    )


def test_load_local_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    _create_plugin(plugin_dir)
    monkeypatch.setenv(PLUGIN_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("importlib.metadata.entry_points", lambda *a, **k: [])
    import compact_memory.plugin_loader as pl

    pl._loaded = False
    _COMPRESSION_REGISTRY.pop("dummy_plugin", None)
    _COMPRESSION_INFO.pop("dummy_plugin", None)
    load_plugins()
    assert "dummy_plugin" in _COMPRESSION_REGISTRY
    info = get_strategy_metadata("dummy_plugin")
    assert info and info["source"].startswith("local")
    _COMPRESSION_REGISTRY.pop("dummy_plugin", None)
    _COMPRESSION_INFO.pop("dummy_plugin", None)
    pl._loaded = False
