import os
import pytest
import shutil
from pathlib import Path
import yaml # PyYAML needed

from gist_memory.config import Config, DEFAULT_CONFIG, ENV_VAR_PREFIX, USER_CONFIG_DIR, USER_CONFIG_PATH, LOCAL_CONFIG_PATH

# Define a fixture for a temporary config directory structure
@pytest.fixture
def temp_config_env(tmp_path, monkeypatch):
    # Standardize user config path for tests based on tmp_path
    # The Config class expands USER_CONFIG_PATH using Path.home()
    # So, we need to ensure Path.home() is patched before USER_CONFIG_PATH is used by Config

    # Monkeypatch home directory first
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Now that home is patched, USER_CONFIG_DIR and USER_CONFIG_PATH in the SUT (config.py)
    # will resolve relative to tmp_path if they are re-evaluated or if Config() re-imports them.
    # To be absolutely sure Config uses the patched home for its path constants,
    # we might need to reload it or pass paths, but Config() should pick it up.

    # Create user config dir structure based on the patched Path.home()
    # USER_CONFIG_DIR is usually ~/.config/gist_memory
    # USER_CONFIG_PATH is USER_CONFIG_DIR / "config.yaml"
    # We use tmp_path as mocked home, so user_config_dir_test becomes tmp_path / ".config" / "gist_memory"

    user_config_dir_test = tmp_path / ".config" / "gist_memory" # This matches how USER_CONFIG_DIR would resolve
    user_config_dir_test.mkdir(parents=True, exist_ok=True)
    user_config_file_test = user_config_dir_test / "config.yaml"

    # Create local project dir: tmp_path / project / .gmconfig.yaml
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    local_config_file_test = project_dir / ".gmconfig.yaml"

    # Monkeypatch current working directory to be the project_dir
    monkeypatch.setattr(os, "getcwd", lambda: str(project_dir))

    # Dynamically get actual USER_CONFIG_PATH and LOCAL_CONFIG_PATH as used by the Config class
    # after monkeypatching, to ensure assertions use the correct paths.
    # This is tricky because those might be module-level constants in config.py.
    # For robust testing, it's better if Config class resolves these paths dynamically or takes them as params.
    # Assuming they are resolved dynamically or the test setup correctly influences them.
    # For this test, we'll rely on the monkeypatched Path.home() being effective for USER_CONFIG_PATH
    # and os.getcwd() for LOCAL_CONFIG_PATH's relative resolution if it's just Path(".gmconfig.yaml").

    # Clear relevant GIST_MEMORY environment variables
    env_vars_to_clear = [
        f"{ENV_VAR_PREFIX}GIST_MEMORY_PATH", # Adjusted to match how Config class generates them
        f"{ENV_VAR_PREFIX}DEFAULT_MODEL_ID",
        f"{ENV_VAR_PREFIX}DEFAULT_STRATEGY_ID",
        f"{ENV_VAR_PREFIX}LOG_FILE",
        f"{ENV_VAR_PREFIX}VERBOSE"
    ]
    original_env_vars = {}
    for var_raw_key in DEFAULT_CONFIG.keys(): # Iterate over known config keys
        var = f"{ENV_VAR_PREFIX}{var_raw_key.upper()}"
        if var not in env_vars_to_clear: # Add any other default keys if not explicitly listed
            env_vars_to_clear.append(var)

    for var in env_vars_to_clear:
        if var in os.environ:
            original_env_vars[var] = os.environ[var]
            monkeypatch.delenv(var, raising=False)
        else:
            original_env_vars[var] = None # To signify it wasn't there initially

    # Yield the paths of the config files created for tests to use
    yield user_config_file_test, local_config_file_test, monkeypatch

    # Teardown: remove temporary directories and restore environment variables
    # No need to manually rmtree tmp_path, pytest handles it.
    # We only created subdirs within tmp_path.
    for var, val in original_env_vars.items():
        if val is not None:
            monkeypatch.setenv(var, val)
        elif var in os.environ: # If it was None (not set) before, ensure it's unset now
             monkeypatch.delenv(var, raising=False)


def test_config_default_values(temp_config_env):
    user_cfg_file, local_cfg_file, _ = temp_config_env
    # Ensure no config files exist yet for this test
    if user_cfg_file.exists(): user_cfg_file.unlink()
    if local_cfg_file.exists(): local_cfg_file.unlink()

    config = Config()
    for key, default_value in DEFAULT_CONFIG.items():
        expected_value = default_value
        if key == "gist_memory_path": # Path.home() is monkeypatched
            expected_value = str(Path(tmp_path) / ".local" / "share" / "gist_memory")
        elif isinstance(default_value, Path):
             expected_value = str(default_value) # Convert Path objects in defaults to string for comparison if needed

        assert config.get(key) == expected_value
        value, source = config.get_with_source(key)
        assert value == expected_value
        assert source == "application default"

def test_config_user_global_file(temp_config_env):
    user_cfg_file, local_cfg_file, _ = temp_config_env
    if local_cfg_file.exists(): local_cfg_file.unlink()

    user_settings = {
        "gist_memory_path": "/user/path/test_global", # Using a more unique path
        "default_model_id": "user_model_global",
        "verbose": True # DEFAULT_CONFIG has 'verbose': False
    }
    with open(user_cfg_file, "w") as f:
        yaml.dump(user_settings, f)

    config = Config() # Config will use the monkeypatched Path.home() to find user_cfg_file

    assert config.get("gist_memory_path") == "/user/path/test_global"
    assert config.get("default_model_id") == "user_model_global"
    assert config.get("verbose") is True
    assert config.get("default_strategy_id") == DEFAULT_CONFIG["default_strategy_id"] # From default

    _, source = config.get_with_source("gist_memory_path")
    # The source description in Config uses the resolved USER_CONFIG_PATH
    # which is now influenced by monkeypatching Path.home()
    expected_user_config_path_str = str(user_cfg_file.resolve())
    assert source == f"user global config file ({expected_user_config_path_str})"
    _, source = config.get_with_source("verbose")
    assert source == f"user global config file ({expected_user_config_path_str})"


def test_config_local_project_file(temp_config_env):
    user_cfg_file, local_cfg_file, _ = temp_config_env
    if user_cfg_file.exists(): user_cfg_file.unlink() # Ensure no user global for this test

    local_settings = {
        "gist_memory_path": "./project_memory_data", # Relative path
        "default_strategy_id": "project_strategy_local"
    }
    with open(local_cfg_file, "w") as f:
        yaml.dump(local_settings, f)

    config = Config() # Config will use monkeypatched os.getcwd() for local file

    # Path should be resolved relative to project_dir (which is os.getcwd())
    project_dir_path = Path(os.getcwd())
    expected_project_path = str((project_dir_path / "project_memory_data").resolve())

    assert config.get("gist_memory_path") == expected_project_path
    assert config.get("default_strategy_id") == "project_strategy_local"

    _, source = config.get_with_source("gist_memory_path")
    expected_local_config_path_str = str(local_cfg_file.resolve())
    assert source == f"local project config file ({expected_local_config_path_str})"


def test_config_environment_variables(temp_config_env):
    user_cfg_file, local_cfg_file, monkeypatch = temp_config_env
    if user_cfg_file.exists(): user_cfg_file.unlink()
    if local_cfg_file.exists(): local_cfg_file.unlink()

    # Use ENV_VAR_PREFIX and .upper() as defined in Config class for env var names
    monkeypatch.setenv(f"{ENV_VAR_PREFIX}GIST_MEMORY_PATH", "/env/path_test")
    monkeypatch.setenv(f"{ENV_VAR_PREFIX}DEFAULT_MODEL_ID", "env_model_test")
    monkeypatch.setenv(f"{ENV_VAR_PREFIX}VERBOSE", "true") # Test string to bool conversion

    config = Config()
    assert config.get("gist_memory_path") == "/env/path_test"
    assert config.get("default_model_id") == "env_model_test"
    assert config.get("verbose") is True # Should be converted to bool

    _, source = config.get_with_source("gist_memory_path")
    assert source == f"environment variable ({ENV_VAR_PREFIX}GIST_MEMORY_PATH)"
    _, source = config.get_with_source("verbose")
    assert source == f"environment variable ({ENV_VAR_PREFIX}VERBOSE)"


def test_config_precedence(temp_config_env):
    user_cfg_file, local_cfg_file, monkeypatch = temp_config_env
    project_dir_path = Path(os.getcwd())


    # 2. User Global Config (will be created at user_cfg_file path)
    with open(user_cfg_file, "w") as f:
        yaml.dump({"gist_memory_path": "/user/path_prec",
                     "default_model_id": "user_model_prec",
                     "verbose": False}, f)

    # 3. Local Project Config (will be created at local_cfg_file path)
    with open(local_cfg_file, "w") as f:
        yaml.dump({"gist_memory_path": "./local/path_prec", # Relative to project_dir
                     "default_model_id": "local_model_prec"}, f) # verbose not set here

    # 4. Environment Variables (highest precedence among these)
    monkeypatch.setenv(f"{ENV_VAR_PREFIX}GIST_MEMORY_PATH", "/env/path_prec")
    # default_model_id not set by env var, so local should take precedence over user

    config = Config()

    expected_local_path_resolved = str((project_dir_path / "./local/path_prec").resolve())
    expected_user_config_path_str = str(user_cfg_file.resolve())
    expected_local_config_path_str = str(local_cfg_file.resolve())


    assert config.get("gist_memory_path") == "/env/path_prec" # Env overrides local and user
    _, source = config.get_with_source("gist_memory_path")
    assert source == f"environment variable ({ENV_VAR_PREFIX}GIST_MEMORY_PATH)"

    assert config.get("default_model_id") == "local_model_prec" # Local overrides user
    _, source = config.get_with_source("default_model_id")
    assert source == f"local project config file ({expected_local_config_path_str})"

    assert config.get("verbose") is False # From user_cfg_file, as local/env didn't set it
    _, source = config.get_with_source("verbose")
    assert source == f"user global config file ({expected_user_config_path_str})"

    # default_strategy_id comes from application default
    assert config.get("default_strategy_id") == DEFAULT_CONFIG["default_strategy_id"]
    _, source = config.get_with_source("default_strategy_id")
    assert source == "application default"


def test_config_set_and_get_all_with_sources(temp_config_env):
    user_cfg_file, local_cfg_file, monkeypatch = temp_config_env
    if local_cfg_file.exists(): local_cfg_file.unlink() # No local override for this test
    if user_cfg_file.exists(): user_cfg_file.unlink() # Start with no user file either

    config = Config() # This will load defaults

    # Set a value, should go to user_cfg_file (path derived from monkeypatched Path.home())
    assert config.set("default_model_id", "new_model_set_via_method") is True
    assert config.set("verbose", "true") is True # Test string to bool conversion via set
    assert config.set("log_file", "/tmp/test.log") is True # Test setting a string that might be None by default

    # Verify it's in the file (user_cfg_file is the path where it should be written)
    assert user_cfg_file.exists()
    user_file_content = yaml.safe_load(user_cfg_file.read_text())
    assert user_file_content["default_model_id"] == "new_model_set_via_method"
    assert user_file_content["verbose"] is True # YAML dumper should handle bool
    assert user_file_content["log_file"] == "/tmp/test.log"


    # Create new config instance to load from the file we just wrote
    config_reloaded = Config()
    assert config_reloaded.get("default_model_id") == "new_model_set_via_method"
    assert config_reloaded.get("verbose") is True
    assert config_reloaded.get("log_file") == "/tmp/test.log"


    # Check sources from the reloaded config
    all_sources = config_reloaded.get_all_with_sources()
    expected_user_config_path_str = str(user_cfg_file.resolve())

    assert all_sources["default_model_id"][0] == "new_model_set_via_method"
    assert all_sources["default_model_id"][1] == f"user global config file ({expected_user_config_path_str})"
    assert all_sources["verbose"][0] is True
    assert all_sources["verbose"][1] == f"user global config file ({expected_user_config_path_str})"
    assert all_sources["log_file"][0] == "/tmp/test.log"
    assert all_sources["log_file"][1] == f"user global config file ({expected_user_config_path_str})"

    # Default value check from reloaded config
    expected_default_path = str(Path(tmp_path) / ".local" / "share" / "gist_memory")
    assert all_sources["gist_memory_path"][0] == expected_default_path
    assert all_sources["gist_memory_path"][1] == "application default"

    # Test setting an invalid key (Config.set should prevent this)
    assert config.set("invalid_key_here", "some_value") is False


def test_config_malformed_yaml(temp_config_env, capsys):
    user_cfg_file, local_cfg_file, _ = temp_config_env
    if local_cfg_file.exists(): local_cfg_file.unlink()

    user_cfg_file.write_text("gist_memory_path: /malformed_path\nverbose: [true") # Malformed YAML

    config = Config() # Should not crash

    # Check that it fell back to defaults
    expected_default_path = str(Path(tmp_path) / ".local" / "share" / "gist_memory")
    assert config.get("gist_memory_path") == expected_default_path
    assert config.get("verbose") == DEFAULT_CONFIG["verbose"]

    # Check for error message (printed by Config class)
    captured = capsys.readouterr()
    assert "Error loading user config" in captured.out or "Error loading user config" in captured.err


def test_config_type_conversion_from_env(temp_config_env):
    user_cfg_file, local_cfg_file, monkeypatch = temp_config_env
    if user_cfg_file.exists(): user_cfg_file.unlink()
    if local_cfg_file.exists(): local_cfg_file.unlink()

    monkeypatch.setenv(f"{ENV_VAR_PREFIX}VERBOSE", "FaLsE") # Mixed case boolean string
    # For a key not in DEFAULT_CONFIG, Config class currently doesn't load it.
    # To test conversion for other types like int, they must be in DEFAULT_CONFIG.
    # Let's assume 'log_file' is string by default, and 'default_retries' is int if added to DEFAULT_CONFIG.
    # DEFAULT_CONFIG["default_retries"] = 3 # Hypothetical
    # monkeypatch.setenv(f"{ENV_VAR_PREFIX}DEFAULT_RETRIES", "5")

    config = Config()
    assert config.get("verbose") is False # Should convert "FaLsE" to False

    # if "default_retries" in DEFAULT_CONFIG:
    #     assert config.get("default_retries") == 5
    #     _, source = config.get_with_source("default_retries")
    #     assert source == f"environment variable ({ENV_VAR_PREFIX}DEFAULT_RETRIES)"
    # del DEFAULT_CONFIG["default_retries"] # Clean up if added
```
