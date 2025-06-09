from pathlib import Path
import pytest
from typer.testing import CliRunner

from compact_memory.cli.main import app
from compact_memory import config as cfg


try:
    runner = CliRunner(mix_stderr=False)
except TypeError:
    runner = CliRunner()


def _patch_cli_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    local_file = tmp_path / ".gmconfig.yaml"
    llm_models_config_file = tmp_path / "llm_models_config.yaml"

    monkeypatch.setattr(cfg, "USER_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "USER_CONFIG_PATH", user_dir / "config.yaml")
    monkeypatch.setattr(cfg, "LOCAL_CONFIG_PATH", local_file)
    monkeypatch.setattr(cfg, "LLM_MODELS_CONFIG_PATH", llm_models_config_file)

    monkeypatch.setattr(
        cfg,
        "SOURCE_USER_CONFIG",
        f"user global config file ({user_dir / 'config.yaml'})",
        raising=False,
    )
    monkeypatch.setattr(
        cfg,
        "SOURCE_LOCAL_CONFIG",
        f"local project config file ({local_file})",
        raising=False,
    )
    monkeypatch.setattr(
        cfg,
        "SOURCE_LLM_MODELS_CONFIG",
        f"llm models config file ({llm_models_config_file})",
        raising=False,
    )

    for key in cfg.DEFAULT_CONFIG:
        monkeypatch.delenv(cfg.ENV_VAR_PREFIX + key.upper(), raising=False)


@pytest.fixture
def patched_cli_config_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_cli_paths(monkeypatch, tmp_path)
    return tmp_path


def test_config_set_and_show(patched_cli_config_paths: Path):
    result = runner.invoke(app, ["config", "set", "default_model_id", "cli_model"])
    assert result.exit_code == 0
    assert "Successfully set" in result.stdout

    show = runner.invoke(app, ["config", "show", "--key", "default_model_id"])
    assert show.exit_code == 0
    assert "cli_model" in show.stdout


def test_config_set_invalid_key(patched_cli_config_paths: Path):
    result = runner.invoke(app, ["config", "set", "unknown_key", "val"])
    assert result.exit_code != 0
    assert "not a recognized" in result.stderr.lower()


def test_config_show_invalid_key(patched_cli_config_paths: Path):
    result = runner.invoke(app, ["config", "show", "--key", "bad_key"])
    assert result.exit_code != 0
    assert "not a recognized key" in result.stderr.lower()
