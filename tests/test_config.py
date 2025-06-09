from pathlib import Path
import yaml
import os
import sys  # For capturing stderr

import pytest

from compact_memory import config as cfg


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    local_file = tmp_path / ".gmconfig.yaml"
    llm_models_config_file = (
        tmp_path / "llm_models_config.yaml"
    )  # New path for LLM models config

    monkeypatch.setattr(cfg, "USER_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "USER_CONFIG_PATH", user_dir / "config.yaml")
    monkeypatch.setattr(cfg, "LOCAL_CONFIG_PATH", local_file)
    monkeypatch.setattr(
        cfg, "LLM_MODELS_CONFIG_PATH", llm_models_config_file
    )  # Patch the new path

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
        "SOURCE_LLM_MODELS_CONFIG",  # Ensure this source string is also updated if used in asserts
        f"llm models config file ({llm_models_config_file})",
        raising=False,
    )

    for key in cfg.DEFAULT_CONFIG:
        monkeypatch.delenv(cfg.ENV_VAR_PREFIX + key.upper(), raising=False)


@pytest.fixture
def patched_config_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):  # Renamed fixture for clarity
    _patch_paths(monkeypatch, tmp_path)
    return tmp_path  # Return tmp_path for creating config files in tests


def test_defaults(patched_config_paths: Path):
    conf = cfg.Config()
    val, src = conf.get_with_source("compact_memory_path")
    assert val == cfg.DEFAULT_CONFIG["compact_memory_path"]
    assert src == cfg.SOURCE_DEFAULT


def test_user_config_override(
    monkeypatch: pytest.MonkeyPatch, patched_config_paths: Path
):
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


def test_local_config_override(
    monkeypatch: pytest.MonkeyPatch, patched_config_paths: Path
):
    local_file = cfg.LOCAL_CONFIG_PATH
    local_file.write_text(
        yaml.dump(
            {"default_model_id": "local/model", "default_engine_id": "local_strategy"}
        )
    )
    conf = cfg.Config()
    assert conf.get("default_model_id") == "local/model"
    assert conf.get_with_source("default_model_id")[1] == cfg.SOURCE_LOCAL_CONFIG
    assert conf.get("default_engine_id") == "local_strategy"
    assert conf.get_with_source("default_engine_id")[1] == cfg.SOURCE_LOCAL_CONFIG


def test_env_var_override(monkeypatch: pytest.MonkeyPatch, patched_config_paths: Path):
    monkeypatch.setenv(cfg.ENV_VAR_PREFIX + "DEFAULT_ENGINE_ID", "env_strategy")
    monkeypatch.setenv(cfg.ENV_VAR_PREFIX + "COMPACT_MEMORY_PATH", "/env/path")
    conf = cfg.Config()
    # Resolve path for comparison
    expected_path = str(Path("/env/path").expanduser().resolve())
    assert conf.get("compact_memory_path") == expected_path
    assert conf.get_with_source("compact_memory_path")[1].startswith(cfg.SOURCE_ENV_VAR)
    assert conf.get("default_engine_id") == "env_strategy"
    assert conf.get_with_source("default_engine_id")[1].startswith(cfg.SOURCE_ENV_VAR)


def test_update_from_cli(patched_config_paths: Path):
    conf = cfg.Config()
    conf.update_from_cli("default_model_id", "cli/model")
    val, src = conf.get_with_source("default_model_id")
    assert val == "cli/model"
    assert src == "command-line argument"


def test_set_and_persist(patched_config_paths: Path):
    conf = cfg.Config()
    assert conf.set("default_model_id", "set/model")
    val, src = conf.get_with_source("default_model_id")
    assert val == "set/model"
    assert src == cfg.SOURCE_OVERRIDE
    # Create a new Config instance to check persistence
    _patch_paths(
        pytest.MonkeyPatch(), patched_config_paths
    )  # Re-patch for new instance
    reload_conf = cfg.Config()
    val2, src2 = reload_conf.get_with_source("default_model_id")
    assert val2 == "set/model"
    assert src2 == cfg.SOURCE_USER_CONFIG


def test_set_invalid_type(patched_config_paths: Path):
    conf = cfg.Config()
    assert not conf.set(
        "default_model_id", 12345
    )  # Attempt to set int where str is expected
    val, src = conf.get_with_source("default_model_id")
    assert val == cfg.DEFAULT_CONFIG["default_model_id"]  # Should remain default
    assert src == cfg.SOURCE_DEFAULT


# --- Tests for LLM Models Config ---


def test_load_llm_models_config_success(patched_config_paths: Path):
    llm_config_file = cfg.LLM_MODELS_CONFIG_PATH
    dummy_llm_configs = {
        "gpt-4-turbo": {"provider": "openai", "api_key": "sk-test", "max_tokens": 8000},
        "local_summarizer": {
            "provider": "local",
            "model_path": "/models/summarizer",
            "device": "cpu",
        },
    }
    llm_config_file.write_text(yaml.dump(dummy_llm_configs))

    conf = cfg.Config()

    assert conf.llm_configs == dummy_llm_configs
    assert conf.get_llm_config("gpt-4-turbo") == dummy_llm_configs["gpt-4-turbo"]
    assert (
        conf.get_llm_config("local_summarizer") == dummy_llm_configs["local_summarizer"]
    )
    assert conf.get_all_llm_configs() == dummy_llm_configs


def test_get_llm_config_non_existent(patched_config_paths: Path):
    llm_config_file = cfg.LLM_MODELS_CONFIG_PATH
    # Create an empty llm_models_config.yaml or one without the requested key
    llm_config_file.write_text(yaml.dump({"existing_model": {"provider": "test"}}))

    conf = cfg.Config()
    assert conf.get_llm_config("non_existent_model") is None


def test_load_llm_models_config_file_not_found(patched_config_paths: Path):
    # Ensure LLM_MODELS_CONFIG_PATH does not exist (it shouldn't by default in tmp_path)
    conf = cfg.Config()
    assert conf.llm_configs == {}  # Should be empty if file not found
    assert conf.get_all_llm_configs() == {}


def test_load_llm_models_config_empty_file(patched_config_paths: Path, capsys):
    llm_config_file = cfg.LLM_MODELS_CONFIG_PATH
    llm_config_file.write_text("")  # Empty file

    conf = cfg.Config()
    assert conf.llm_configs == {}

    # Check for warning about non-dictionary content
    # The current implementation logs if loaded_llm_configs is not None but also not a dict.
    # An empty file results in loaded_llm_configs being None, so no warning for "empty".
    # If the file had content like "null" or "- item", then it would warn.
    # For truly empty file, no warning is expected by current code.

    llm_config_file.write_text("null")  # content that is not a dict
    conf_null = cfg.Config()
    assert conf_null.llm_configs == {}
    captured = capsys.readouterr()
    assert "does not contain a valid dictionary" in captured.err


def test_load_llm_models_config_malformed_yaml(patched_config_paths: Path, capsys):
    llm_config_file = cfg.LLM_MODELS_CONFIG_PATH
    llm_config_file.write_text(
        "gpt-4-turbo: {provider: openai\n  api_key: sk-test"
    )  # Malformed YAML

    conf = cfg.Config()
    assert conf.llm_configs == {}  # Should remain empty on error

    captured = capsys.readouterr()
    assert "Error parsing LLM models config file" in captured.err
    assert str(llm_config_file) in captured.err


def test_config_validate_and_get_all(patched_config_paths: Path):
    conf = cfg.Config()
    assert conf.validate()
    all_data = conf.get_all_with_sources()
    for key in cfg.DEFAULT_CONFIG:
        assert key in all_data
        assert all_data[key][1] in (
            cfg.SOURCE_DEFAULT,
            cfg.SOURCE_USER_CONFIG,
            cfg.SOURCE_LOCAL_CONFIG,
            cfg.SOURCE_ENV_VAR + f" ({cfg.ENV_VAR_PREFIX + key.upper()})",
            cfg.SOURCE_OVERRIDE,
        )
