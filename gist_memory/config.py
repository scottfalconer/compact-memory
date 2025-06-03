import os
import pathlib
import yaml
from typing import Any, Dict, Optional

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

# Environment variable prefixes
ENV_VAR_PREFIX = "GIST_MEMORY_"

class Config:
    def __init__(self):
        self._config = {}  # Initialize as empty, precedence will fill it
        self._load_defaults()
        self._load_user_config()
        self._load_local_config()
        self._load_env_vars()

    def _load_defaults(self):
        self._config.update(DEFAULT_CONFIG)

    def _load_user_config(self):
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH, "r") as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        self._config.update(user_config)
            except Exception as e:
                print(f"Error loading user config: {e}")


    def _load_local_config(self):
        if LOCAL_CONFIG_PATH.exists():
            try:
                with open(LOCAL_CONFIG_PATH, "r") as f:
                    local_config = yaml.safe_load(f)
                    if local_config:
                        self._config.update(local_config)
            except Exception as e:
                print(f"Error loading local config: {e}")

    def _load_env_vars(self):
        for key in DEFAULT_CONFIG.keys(): # Iterate over known config keys
            env_var_name = ENV_VAR_PREFIX + key.upper()
            env_var_value = os.getenv(env_var_name)
            if env_var_value:
                # Attempt to cast env var value to the type of the default value
                default_value = DEFAULT_CONFIG[key]
                try:
                    if isinstance(default_value, bool):
                        env_var_value = env_var_value.lower() in ("true", "1", "yes")
                    else:
                        env_var_value = type(default_value)(env_var_value)
                    self._config[key] = env_var_value
                except ValueError:
                    print(f"Warning: Could not cast env var {env_var_name} to type {type(default_value)}. Using string value.")
                    self._config[key] = env_var_value


    def get(self, key: str) -> Any:
        return self._config.get(key)

    def set(self, key: str, value: Any) -> None:
        # This method should primarily update the user global config.
        # The runtime config (`self._config`) is already updated by the caller or will be on next load.

        # Validate before setting
        if key in DEFAULT_CONFIG:
            default_value = DEFAULT_CONFIG[key]
            if not isinstance(value, type(default_value)):
                try:
                    value = type(default_value)(value)
                except ValueError:
                    print(f"Error: Invalid type for '{key}'. Expected {type(default_value)}, got {type(value)}.")
                    return

        # Load current user config
        user_config_data = {}
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH, "r") as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        user_config_data = loaded_config
            except Exception as e:
                print(f"Error reading user config before set: {e}")

        # Update the specific key-value pair
        user_config_data[key] = value

        # Write back to user global config
        try:
            USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(USER_CONFIG_PATH, "w") as f:
                yaml.dump(user_config_data, f)
            # Update runtime config immediately for consistency if needed by application logic
            self._config[key] = value
        except Exception as e:
            print(f"Error writing to user config: {e}")


    def validate(self) -> bool:
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in self._config:
                print(f"Validation Error: Missing configuration key: {key}")
                return False

            current_value = self._config[key]
            if not isinstance(current_value, type(default_value)):
                # Attempt type coercion for strings from env vars, if they are compatible
                if isinstance(current_value, str):
                    try:
                        if isinstance(default_value, bool):
                             #This case should ideally be handled during load_env_vars
                            coerced_value = current_value.lower() in ('true', '1', 'yes')
                        else:
                            coerced_value = type(default_value)(current_value)

                        if not isinstance(coerced_value, type(default_value)):
                           raise ValueError("Coercion did not result in the correct type")
                        self._config[key] = coerced_value # Update with coerced value
                    except ValueError:
                        print(f"Validation Error: Configuration key '{key}' has incorrect type. Expected {type(default_value)}, got {type(current_value)}.")
                        return False
                else:
                    print(f"Validation Error: Configuration key '{key}' has incorrect type. Expected {type(default_value)}, got {type(current_value)}.")
                    return False
        return True

if __name__ == "__main__":
    # Test basic loading and precedence (manual setup for testing)
    print(f"Default config: {DEFAULT_CONFIG}")

    # Create dummy user config
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG_PATH, "w") as f:
        yaml.dump({"gist_memory_path": "/user/path", "default_model_id": "user/model"}, f)

    # Create dummy local config
    with open(LOCAL_CONFIG_PATH, "w") as f:
        yaml.dump({"default_model_id": "local/model", "default_strategy_id": "local_strategy"}, f)

    # Set environment variables
    os.environ[ENV_VAR_PREFIX + "DEFAULT_STRATEGY_ID"] = "env_strategy"
    os.environ[ENV_VAR_PREFIX + "GIST_MEMORY_PATH"] = "/env/path"


    config = Config()
    print(f"\nLoaded config object: {config._config}")
    print(f"GIST_MEMORY_PATH: {config.get('gist_memory_path')}") # Expected: /env/path (Env)
    print(f"DEFAULT_MODEL_ID: {config.get('default_model_id')}") # Expected: local/model (Local)
    print(f"DEFAULT_STRATEGY_ID: {config.get('default_strategy_id')}") # Expected: env_strategy (Env)

    print(f"\nValidation result before set: {config.validate()}")

    # Test set
    print("\nTesting set function...")
    config.set("default_model_id", "new_user_model_via_set")

    # Verify runtime config updated
    print(f"Config after set (runtime): {config.get('default_model_id')}")

    # Verify user config file updated
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, "r") as f:
            user_conf_after_set = yaml.safe_load(f)
            print(f"User config file after set: {user_conf_after_set}")
    else:
        print("User config file not found after set.")

    config.set("new_setting", "test_value") # Test setting a new key
    print(f"Config after set (new_setting): {config.get('new_setting')}")
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, "r") as f:
            user_conf_after_set_new = yaml.safe_load(f)
            print(f"User config file after set (new_setting): {user_conf_after_set_new}")

    print(f"\nValidation result after set: {config.validate()}")

    # Clean up dummy files and env vars
    if USER_CONFIG_PATH.exists():
        os.remove(USER_CONFIG_PATH)
    if LOCAL_CONFIG_PATH.exists():
        os.remove(LOCAL_CONFIG_PATH)
    del os.environ[ENV_VAR_PREFIX + "DEFAULT_STRATEGY_ID"]
    del os.environ[ENV_VAR_PREFIX + "GIST_MEMORY_PATH"]

    print("\nTesting with a missing key for validation...")
    temp_config = Config()
    del temp_config._config['default_model_id'] # Manually remove a key
    print(f"Validation with missing key: {temp_config.validate()}")

    print("\nTesting with a type mismatch for validation...")
    temp_config_type = Config()
    temp_config_type._config['default_model_id'] = 123 # Introduce type error
    print(f"Validation with type mismatch: {temp_config_type.validate()}")
