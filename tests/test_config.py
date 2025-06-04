from pathlib import Path
import yaml

import pytest

from compact_memory import config as cfg


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    local_file = tmp_path / ".gmconfig.yaml"
    monkeypatch.setattr(cfg, "USER_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "USER_CONFIG_PATH", user_dir / "config.yaml")
    monkeypatch.setattr(cfg, "LOCAL_CONFIG_PATH", local_file)
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
    for key in cfg.DEFAULT_CONFIG:
        monkeypatch.delenv(cfg.ENV_VAR_PREFIX + key.upper(), raising=False)


@pytest.fixture
def patched_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_paths(monkeypatch, tmp_path)
    return tmp_path


def test_defaults(patched_config: Path):
    conf = cfg.Config()
    val, src = conf.get_with_source("compact_memory_path")
    assert val == cfg.DEFAULT_CONFIG["compact_memory_path"]
    assert src == cfg.SOURCE_DEFAULT


def test_user_config_override(monkeypatch: pytest.MonkeyPatch, patched_config: Path):
    user_file = cfg.USER_CONFIG_PATH
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text(
        yaml.dump(
            {"compact_memory_path": "/user/path", "default_model_id": "user/model"}
        )
    )
    conf = cfg.Config()
    assert conf.get("compact_memory_path") == "/user/path"
    assert conf.get_with_source("compact_memory_path")[1] == cfg.SOURCE_USER_CONFIG
    assert conf.get("default_model_id") == "user/model"
    assert conf.get_with_source("default_model_id")[1] == cfg.SOURCE_USER_CONFIG


def test_local_config_override(monkeypatch: pytest.MonkeyPatch, patched_config: Path):
    local_file = cfg.LOCAL_CONFIG_PATH
    local_file.write_text(
        yaml.dump(
            {"default_model_id": "local/model", "default_strategy_id": "local_strategy"}
        )
    )
    conf = cfg.Config()
    assert conf.get("default_model_id") == "local/model"
    assert conf.get_with_source("default_model_id")[1] == cfg.SOURCE_LOCAL_CONFIG
    assert conf.get("default_strategy_id") == "local_strategy"
    assert conf.get_with_source("default_strategy_id")[1] == cfg.SOURCE_LOCAL_CONFIG


def test_env_var_override(monkeypatch: pytest.MonkeyPatch, patched_config: Path):
    monkeypatch.setenv(cfg.ENV_VAR_PREFIX + "DEFAULT_STRATEGY_ID", "env_strategy")
    monkeypatch.setenv(cfg.ENV_VAR_PREFIX + "COMPACT_MEMORY_PATH", "/env/path")
    conf = cfg.Config()
    assert conf.get("compact_memory_path") == "/env/path"
    assert conf.get_with_source("compact_memory_path")[1].startswith(cfg.SOURCE_ENV_VAR)
    assert conf.get("default_strategy_id") == "env_strategy"
    assert conf.get_with_source("default_strategy_id")[1].startswith(cfg.SOURCE_ENV_VAR)


def test_update_from_cli(patched_config: Path):
    conf = cfg.Config()
    conf.update_from_cli("default_model_id", "cli/model")
    val, src = conf.get_with_source("default_model_id")
    assert val == "cli/model"
    assert src == "command-line argument"


def test_set_and_persist(patched_config: Path):
    conf = cfg.Config()
    assert conf.set("default_model_id", "set/model")
    val, src = conf.get_with_source("default_model_id")
    assert val == "set/model"
    assert src == cfg.SOURCE_OVERRIDE
    reload_conf = cfg.Config()
    val2, src2 = reload_conf.get_with_source("default_model_id")
    assert val2 == "set/model"
    assert src2 == cfg.SOURCE_USER_CONFIG


def test_set_invalid_type(patched_config: Path):
    conf = cfg.Config()
    assert not conf.set("default_model_id", 12345)
    val, src = conf.get_with_source("default_model_id")
    assert val == cfg.DEFAULT_CONFIG["default_model_id"]
    assert src == cfg.SOURCE_DEFAULT
