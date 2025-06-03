import os
import pathlib
import yaml
from typing import Any, Dict, Optional, Tuple

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "gist_memory_path": "~/.local/share/gist_memory",
    "default_model_id": "openai/gpt-3.5-turbo",
    "default_strategy_id": "default",
}

# Configuration file paths
USER_CONFIG_DIR = pathlib.Path("~/.config/gist_memory").expanduser()
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"
LOCAL_CONFIG_PATH = pathlib.Path(".gmconfig.yaml")

# Source descriptions - MOVED AFTER PATH DEFINITIONS
SOURCE_DEFAULT = "application default"
SOURCE_USER_CONFIG = f"user config ({USER_CONFIG_PATH})"
SOURCE_LOCAL_CONFIG = f"local config ({LOCAL_CONFIG_PATH})"
SOURCE_ENV_VAR = "environment variable"
SOURCE_OVERRIDE = "runtime override" # For values set via CLI options or future `config.set_runtime()`

# Environment variable prefixes
ENV_VAR_PREFIX = "GIST_MEMORY_"

class Config:
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}

        self._load_defaults()
        self._load_user_config()
        self._load_local_config()
        self._load_env_vars()
        # Note: CLI overrides will be handled by the CLI directly by updating self._config and self._sources

    def _load_defaults(self):
        for key, value in DEFAULT_CONFIG.items():
            self._config[key] = value
            self._sources[key] = SOURCE_DEFAULT

    def _load_user_config(self):
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH, "r") as f:
                    user_config = yaml.safe_load(f)
                    if user_config and isinstance(user_config, dict):
                        for key, value in user_config.items():
                            if key in DEFAULT_CONFIG: # Check against DEFAULT_CONFIG to ensure key is known
                                self._config[key] = value
                                self._sources[key] = SOURCE_USER_CONFIG
                            # else: print(f"Warning: Unknown key '{key}' in user config.")
            except Exception as e:
                print(f"Error loading user config: {e}")

    def _load_local_config(self):
        if LOCAL_CONFIG_PATH.exists():
            try:
                with open(LOCAL_CONFIG_PATH, "r") as f:
                    local_config = yaml.safe_load(f)
                    if local_config and isinstance(local_config, dict):
                        for key, value in local_config.items():
                             if key in DEFAULT_CONFIG: # Check against DEFAULT_CONFIG
                                self._config[key] = value
                                self._sources[key] = SOURCE_LOCAL_CONFIG
                            # else: print(f"Warning: Unknown key '{key}' in local config.")
            except Exception as e:
                print(f"Error loading local config: {e}")

    def _load_env_vars(self):
        for key in DEFAULT_CONFIG.keys():
            env_var_name = ENV_VAR_PREFIX + key.upper()
            env_var_value_str = os.getenv(env_var_name)
            if env_var_value_str is not None:
                default_value = DEFAULT_CONFIG[key]
                try:
                    if isinstance(default_value, bool):
                        actual_value = env_var_value_str.lower() in ("true", "1", "yes")
                    elif isinstance(default_value, int): # Added int casting
                        actual_value = int(env_var_value_str)
                    elif isinstance(default_value, float): # Added float casting
                        actual_value = float(env_var_value_str)
                    else:
                        actual_value = env_var_value_str # Assume string

                    self._config[key] = actual_value
                    self._sources[key] = SOURCE_ENV_VAR
                except ValueError:
                    print(f"Warning: Could not cast env var {env_var_name} value '{env_var_value_str}' to type {type(default_value)}. Using string value.")
                    self._config[key] = env_var_value_str
                    self._sources[key] = SOURCE_ENV_VAR


    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any, source: str = SOURCE_OVERRIDE) -> bool:
        """
        Sets a configuration value in the runtime config and persists it to the user global config.
        Source can be specified, defaults to runtime override.
        Returns True if successful, False otherwise.
        """
        original_type = None
        if key in DEFAULT_CONFIG:
            original_type = type(DEFAULT_CONFIG[key])
            # Attempt to cast the input string value to the original type
            if isinstance(value, str): # All values from CLI Argument are strings initially
                try:
                    if original_type is bool:
                        value = value.lower() in ("true", "1", "yes", "on")
                    elif original_type is int:
                        value = int(value)
                    elif original_type is float:
                        value = float(value)
                    # Add other type conversions if necessary
                    # If original_type is str, no conversion needed for string input
                except ValueError:
                    print(f"Error: Invalid value format for '{key}'. Cannot convert '{value}' to {original_type}.")
                    return False # Indicate failure
            elif not isinstance(value, original_type): # If not a string and not the right type
                 print(f"Error: Invalid type for '{key}'. Expected {original_type}, got {type(value)}.")
                 return False # Indicate failure
        # If key is not in DEFAULT_CONFIG, we allow setting it but without type validation based on defaults.

        self._config[key] = value
        self._sources[key] = source # Runtime source

        # Persist to user global config file
        user_config_data = {}
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH, "r") as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        user_config_data = loaded_config
            except Exception as e:
                print(f"Error reading user config before set: {e}")

        user_config_data[key] = value

        try:
            USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(USER_CONFIG_PATH, "w") as f:
                yaml.dump(user_config_data, f)
            return True # Indicate success
        except Exception as e:
            print(f"Error writing to user config: {e}")
            return False # Indicate failure

    def get_with_source(self, key: str) -> Optional[Tuple[Any, str]]:
        if key in self._config:
            return self._config[key], self._sources.get(key, "Unknown")
        elif key in DEFAULT_CONFIG: # Should be caught by _load_defaults
             return DEFAULT_CONFIG[key], SOURCE_DEFAULT
        return None

    def get_all_with_sources(self) -> Dict[str, Tuple[Any, str]]:
        all_data = {}
        # Start with all known default keys
        for key in DEFAULT_CONFIG.keys():
            all_data[key] = (self._config.get(key, DEFAULT_CONFIG[key]),
                             self._sources.get(key, SOURCE_DEFAULT))
        # Add any other keys that might have been set dynamically (e.g. from config files not in defaults)
        # This part is tricky if we strictly only want known keys.
        # For now, let's assume _config only contains keys that were either in DEFAULT_CONFIG or added via set()
        # and set() should ideally only allow known keys or handle them carefully.
        # The current implementation will show all items in self._config due to the loading logic.
        for key_present in self._config:
            if key_present not in all_data: # Should not happen if loading logic is strict to DEFAULT_CONFIG
                 all_data[key_present] = (self._config[key_present], self._sources.get(key_present, SOURCE_OVERRIDE))
        return all_data

    def update_from_cli(self, key: str, value: Any):
        if value is not None:
            original_type = None
            if key in DEFAULT_CONFIG:
                original_type = type(DEFAULT_CONFIG[key])
                if not isinstance(value, original_type): # CLI values are often strings
                    try:
                        if original_type is bool: value = str(value).lower() in ("true", "1", "yes")
                        else: value = original_type(value)
                    except ValueError:
                        print(f"Warning: CLI value for '{key}' ('{value}') could not be cast to {original_type}. Using as is.")

            self._config[key] = value
            self._sources[key] = "command-line argument"


    def validate(self) -> bool: # Mostly for type validation after all loads
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in self._config:
                print(f"Validation Error: Missing configuration key: {key}")
                return False

            current_value = self._config[key]
            expected_type = type(default_value)

            if not isinstance(current_value, expected_type):
                try:
                    if expected_type is bool: current_value = str(current_value).lower() in ('true', '1', 'yes', 'on')
                    elif expected_type is int : current_value = int(str(current_value))
                    elif expected_type is float: current_value = float(str(current_value))
                    else: current_value = expected_type(str(current_value)) # General attempt

                    if isinstance(current_value, expected_type):
                        self._config[key] = current_value
                    else: raise ValueError("Coercion failed.")
                except ValueError:
                    print(f"Validation Error: Key '{key}' has value '{self._config[key]}' of type {type(self._config[key])}, expected {expected_type}.")
                    return False
        return True

if __name__ == "__main__":
    print(f"Default config: {DEFAULT_CONFIG}")

    test_files_to_clean = [USER_CONFIG_PATH, LOCAL_CONFIG_PATH]
    for tf in test_files_to_clean:
        if tf.exists(): os.remove(tf)

    env_vars_to_clean = [ENV_VAR_PREFIX + k.upper() for k in DEFAULT_CONFIG.keys()]
    for ev in env_vars_to_clean:
        if ev in os.environ: del os.environ[ev]

    print("\n--- Test 1: Defaults ---")
    config1 = Config()
    all_conf1 = config1.get_all_with_sources()
    for k, (v, s) in all_conf1.items(): print(f"{k}: {v} (Source: {s})")
    assert config1.get("gist_memory_path") == DEFAULT_CONFIG["gist_memory_path"]
    assert config1.get_with_source("gist_memory_path")[1] == SOURCE_DEFAULT

    print("\n--- Test 2: User Config Override ---")
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG_PATH, "w") as f: yaml.dump({"gist_memory_path": "/user/path", "default_model_id": "user/model"}, f)
    config2 = Config()
    for k, (v, s) in config2.get_all_with_sources().items(): print(f"{k}: {v} (Source: {s})")
    assert config2.get("gist_memory_path") == "/user/path"; assert config2.get_with_source("gist_memory_path")[1] == SOURCE_USER_CONFIG
    assert config2.get("default_model_id") == "user/model"; assert config2.get_with_source("default_model_id")[1] == SOURCE_USER_CONFIG

    print("\n--- Test 3: Local Config Override ---")
    with open(LOCAL_CONFIG_PATH, "w") as f: yaml.dump({"default_model_id": "local/model", "default_strategy_id": "local_strategy"}, f)
    config3 = Config()
    for k, (v, s) in config3.get_all_with_sources().items(): print(f"{k}: {v} (Source: {s})")
    assert config3.get("default_model_id") == "local/model"; assert config3.get_with_source("default_model_id")[1] == SOURCE_LOCAL_CONFIG
    assert config3.get("default_strategy_id") == "local_strategy"; assert config3.get_with_source("default_strategy_id")[1] == SOURCE_LOCAL_CONFIG

    print("\n--- Test 4: Env Var Override ---")
    os.environ[ENV_VAR_PREFIX + "DEFAULT_STRATEGY_ID"] = "env_strategy"
    os.environ[ENV_VAR_PREFIX + "GIST_MEMORY_PATH"] = "/env/path"
    config4 = Config()
    for k, (v, s) in config4.get_all_with_sources().items(): print(f"{k}: {v} (Source: {s})")
    assert config4.get("gist_memory_path") == "/env/path"; assert config4.get_with_source("gist_memory_path")[1] == SOURCE_ENV_VAR
    assert config4.get("default_strategy_id") == "env_strategy"; assert config4.get_with_source("default_strategy_id")[1] == SOURCE_ENV_VAR

    print("\n--- Test 5: CLI Update (Simulated) ---")
    config_cli = Config() # Fresh config
    config_cli.update_from_cli("default_model_id", "cli/model")
    cli_val, cli_src = config_cli.get_with_source("default_model_id")
    print(f"default_model_id: {cli_val} (Source: {cli_src})")
    assert cli_val == "cli/model"; assert cli_src == "command-line argument"
    # Test CLI override for a boolean
    config_cli.update_from_cli("some_bool_not_in_defaults_yet", "true") # Assuming it might be added later
    # This will be added as a new key with "command-line argument" source

    print("\n--- Test 6: Set and Persist (User Global) ---")
    if USER_CONFIG_PATH.exists(): os.remove(USER_CONFIG_PATH) # Clean user config
    if LOCAL_CONFIG_PATH.exists(): os.remove(LOCAL_CONFIG_PATH) # Clean local
    if ENV_VAR_PREFIX + "DEFAULT_MODEL_ID" in os.environ: del os.environ[ENV_VAR_PREFIX + "DEFAULT_MODEL_ID"]

    config_for_set = Config() # Should load defaults
    print("Setting 'default_model_id' to 'set/model'")
    success = config_for_set.set("default_model_id", "set/model")
    assert success
    val_set, src_set = config_for_set.get_with_source("default_model_id")
    print(f"Runtime after set: default_model_id = {val_set} (Source: {src_set})")
    assert val_set == "set/model"; assert src_set == SOURCE_OVERRIDE

    config_reloaded = Config() # New instance, should pick up from user config
    val_reloaded, src_reloaded = config_reloaded.get_with_source("default_model_id")
    print(f"Reloaded after set: default_model_id = {val_reloaded} (Source: {src_reloaded})")
    assert val_reloaded == "set/model"; assert src_reloaded == SOURCE_USER_CONFIG

    print("\n--- Test 7: Set invalid key type ---")
    # Assuming default_model_id is string, try setting an int
    success_invalid_type = config_for_set.set("default_model_id", 12345)
    assert not success_invalid_type
    val_invalid, src_invalid = config_for_set.get_with_source("default_model_id")
    print(f"After invalid set: default_model_id = {val_invalid} (Source: {src_invalid})")
    assert val_invalid == "set/model" # Should remain unchanged

    # Clean up
    if USER_CONFIG_PATH.exists(): os.remove(USER_CONFIG_PATH)
    if LOCAL_CONFIG_PATH.exists(): os.remove(LOCAL_CONFIG_PATH)
    env_vars_to_clean = [ENV_VAR_PREFIX + k.upper() for k in DEFAULT_CONFIG.keys()]
    for ev in env_vars_to_clean:
        if ev in os.environ: del os.environ[ev]

    print("\nAll tests seem to pass based on assertions and printed output.")
