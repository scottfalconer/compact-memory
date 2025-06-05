from pathlib import Path

import pytest

from CompressionEngine.core.registry import ( # Updated path
    _COMPRESSION_REGISTRY,
    _COMPRESSION_INFO,
    get_engine_metadata, # Updated function name
)
from compact_memory.plugin_loader import load_plugins, PLUGIN_ENV_VAR # Assuming PLUGIN_ENV_VAR might need update if related to strategy
from CompressionEngine.core.engines_abc import ( # Updated path
    CompressionEngine, # Updated class name
    CompressedMemory,
    CompressionTrace,
)


def _create_plugin(pkg_dir: Path) -> None:
    pkg_dir.mkdir()
    (pkg_dir / "engine.py").write_text( # Updated filename
        """
from CompressionEngine.core.engines_abc import ( # Updated path
    CompressionEngine, # Updated class name
    CompressedMemory,
    CompressionTrace,
)

class DummyPluginEngine(CompressionEngine): # Updated class name
    id = 'dummy_plugin_engine' # Updated id

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        return CompressedMemory(text='x'), CompressionTrace(
            engine_name=self.id, # Updated parameter name
            engine_params={}, # Updated parameter name
            input_summary={},
            steps=[],
        )
"""
    )
    (pkg_dir / "engine_package.yaml").write_text( # Updated filename
        """
package_format_version: '1.0'
engine_id: dummy_plugin_engine # Updated field name and value
engine_class_name: DummyPluginEngine # Updated field name and value
engine_module: engine # Updated field value
display_name: Dummy Plugin Engine # Updated value
version: '0.1'
authors: ['Tester']
description: Test plugin engine # Updated value
"""
    )


def test_load_local_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    _create_plugin(plugin_dir)
    monkeypatch.setenv(PLUGIN_ENV_VAR, str(tmp_path)) # PLUGIN_ENV_VAR might need to be COMPACT_MEMORY_ENGINES_PATH
    monkeypatch.setattr("importlib.metadata.entry_points", lambda *a, **k: [])
    import compact_memory.plugin_loader as pl

    pl._loaded = False
    _COMPRESSION_REGISTRY.pop("dummy_plugin_engine", None) # Updated key
    _COMPRESSION_INFO.pop("dummy_plugin_engine", None) # Updated key
    load_plugins()
    assert "dummy_plugin_engine" in _COMPRESSION_REGISTRY # Updated key
    info = get_engine_metadata("dummy_plugin_engine") # Updated function call and key
    assert info and info["source"].startswith("local")
    _COMPRESSION_REGISTRY.pop("dummy_plugin_engine", None) # Updated key
    _COMPRESSION_INFO.pop("dummy_plugin_engine", None) # Updated key
    pl._loaded = False


def test_load_entrypoint_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEPEngine(CompressionEngine): # Updated class name
        id = "dummy_ep_engine" # Updated id

        def compress(self, text_or_chunks, llm_token_budget, **kwargs):
            return CompressedMemory(text="y"), CompressionTrace(
                engine_name=self.id, # Updated parameter name
                engine_params={}, # Updated parameter name
                input_summary={},
                steps=[],
            )

    class DummyEP:
        def __init__(self, value: str, obj: type[CompressionEngine]): # Updated type hint
            self.value = value
            self._obj = obj
            self.dist = type(
                "Dist", (), {"metadata": {"Name": "dummy_engine"}, "version": "0.1"} # Updated name
            )()

        def load(self):  # pragma: no cover - executed during test
            return self._obj

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        # Entry point group should likely change from compact_memory.strategies to compact_memory.engines
        lambda *a, **k: [DummyEP("dummy_engine:DummyEPEngine", DummyEPEngine)], # Updated values
    )
    monkeypatch.setattr(
        "compact_memory.plugin_loader._iter_local_plugin_paths", lambda: []
    )
    import compact_memory.plugin_loader as pl

    pl._loaded = False
    _COMPRESSION_REGISTRY.pop("dummy_ep_engine", None) # Updated key
    _COMPRESSION_INFO.pop("dummy_ep_engine", None) # Updated key
    load_plugins() # This function might need to look for a different entry point group
    assert "dummy_ep_engine" in _COMPRESSION_REGISTRY # Updated key
    info = get_engine_metadata("dummy_ep_engine") # Updated function call and key
    assert info and info["source"].startswith("plugin")
    _COMPRESSION_REGISTRY.pop("dummy_ep_engine", None) # Updated key
    _COMPRESSION_INFO.pop("dummy_ep_engine", None) # Updated key
    pl._loaded = False
