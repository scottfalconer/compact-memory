import pytest
from pathlib import Path

from compact_memory.engine_registry import (
    _ENGINE_REGISTRY,
    _ENGINE_INFO,
    get_engine_metadata,
)
from compact_memory.plugin_loader import load_plugins, PLUGIN_ENV_VAR
from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)


def _create_plugin(pkg_dir: Path) -> None:
    pkg_dir.mkdir()
    (pkg_dir / "engine.py").write_text(
        """
from compact_memory.engines import BaseCompressionEngine, CompressedMemory, CompressionTrace

class DummyPluginEngine(BaseCompressionEngine):
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
    (pkg_dir / "engine_package.yaml").write_text(
        """
package_format_version: '1.0'
engine_id: dummy_plugin
engine_class_name: DummyPluginEngine
engine_module: engine
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
    _ENGINE_REGISTRY.pop("dummy_plugin", None)
    _ENGINE_INFO.pop("dummy_plugin", None)
    load_plugins()
    assert "dummy_plugin" in _ENGINE_REGISTRY
    info = get_engine_metadata("dummy_plugin")
    assert info and info["source"].startswith("local")
    _ENGINE_REGISTRY.pop("dummy_plugin", None)
    _ENGINE_INFO.pop("dummy_plugin", None)
    pl._loaded = False


def test_load_entrypoint_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEPEngine(BaseCompressionEngine):
        id = "dummy_ep"

        def compress(self, text_or_chunks, llm_token_budget, **kwargs):
            return CompressedMemory(text="y"), CompressionTrace(
                strategy_name=self.id,
                strategy_params={},
                input_summary={},
                steps=[],
            )

    class DummyEP:
        def __init__(self, value: str, obj: type[BaseCompressionEngine]):
            self.value = value
            self._obj = obj
            self.dist = type(
                "Dist", (), {"metadata": {"Name": "dummy"}, "version": "0.1"}
            )()

        def load(self):  # pragma: no cover - executed during test
            return self._obj

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda *a, **k: [DummyEP("dummy:DummyEPEngine", DummyEPEngine)],
    )
    monkeypatch.setattr(
        "compact_memory.plugin_loader._iter_local_plugin_paths", lambda: []
    )
    import compact_memory.plugin_loader as pl

    pl._loaded = False
    _ENGINE_REGISTRY.pop("dummy_ep", None)
    _ENGINE_INFO.pop("dummy_ep", None)
    load_plugins()
    assert "dummy_ep" in _ENGINE_REGISTRY
    info = get_engine_metadata("dummy_ep")
    assert info and info["source"].startswith("plugin")
    _ENGINE_REGISTRY.pop("dummy_ep", None)
    _ENGINE_INFO.pop("dummy_ep", None)
    pl._loaded = False
