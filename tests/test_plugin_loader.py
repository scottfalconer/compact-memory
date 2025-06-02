import sys
from importlib.metadata import EntryPoint
import yaml
from typer.testing import CliRunner

from gist_memory.cli import app
from gist_memory import plugin_loader
from gist_memory.compression import get_compression_strategy, get_strategy_metadata


def _write_local_plugin(path, name, text):
    path.mkdir(parents=True)
    (path / "strategy.py").write_text(text)
    manifest = {
        "package_format_version": "1.0",
        "strategy_id": name,
        "strategy_class_name": "LocalStrategy",
        "strategy_module": "strategy",
        "display_name": "LocalStrategy",
        "version": "0.0.1",
        "authors": [],
        "description": "test",
    }
    (path / "strategy_package.yaml").write_text(yaml.safe_dump(manifest))
    (path / "requirements.txt").write_text("\n")
    (path / "README.md").write_text("hi")


def test_plugin_override_chain(tmp_path, monkeypatch):
    ep_mod = tmp_path / "epmod.py"
    ep_mod.write_text(
        "from gist_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace\n"
        "class EpStrategy(CompressionStrategy):\n"
        "    id='none'\n"
        "    display_name='EP'\n"
        "    def compress(self, t, llm_token_budget, **k):\n"
        "        return CompressedMemory(text='ep'), CompressionTrace()\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    ep = EntryPoint(name="ep", value="epmod:EpStrategy", group=plugin_loader.ENTRYPOINT_GROUP)
    monkeypatch.setattr(
        plugin_loader.metadata,
        "entry_points",
        lambda group=None: [ep] if group == plugin_loader.ENTRYPOINT_GROUP else [],
    )

    local_root = tmp_path / "local"
    plugin_dir = local_root / "plugin"
    _write_local_plugin(
        plugin_dir,
        "none",
        "from gist_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace\n"
        "class LocalStrategy(CompressionStrategy):\n"
        "    id='none'\n"
        "    display_name='Local'\n"
        "    def compress(self, t, llm_token_budget, **k):\n"
        "        return CompressedMemory(text='local'), CompressionTrace()\n",
    )
    monkeypatch.setenv(plugin_loader.PLUGIN_ENV_VAR, str(local_root))

    plugin_loader._loaded = False
    plugin_loader.load_plugins()

    cls = get_compression_strategy("none")
    assert cls.__name__ == "LocalStrategy"
    info = get_strategy_metadata("none")
    assert info["source"].startswith("local")
    assert info["overrides"] == "plugin (unknown)"

    runner = CliRunner()
    result = runner.invoke(app, ["strategy", "list"])
    assert result.exit_code == 0
    assert "Local" in result.stdout
