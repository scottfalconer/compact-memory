import os
import pathlib
import sys
import yaml
from typing import Any, Dict, Optional, Tuple

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "compact_memory_path": "~/.local/share/compact_memory",
    "default_model_id": "openai/gpt-3.5-turbo",
    "default_engine_id": "no_compression_engine", # Assuming 'no_compression_engine' is a sensible default
    "verbose": False,
    "log_file": None,
}

# Configuration file paths
USER_CONFIG_DIR = pathlib.Path("~/.config/compact_memory").expanduser()
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"
LOCAL_CONFIG_PATH = pathlib.Path(".gmconfig.yaml")

# Source descriptions - MOVED AFTER PATH DEFINITIONS
SOURCE_DEFAULT = "application default"
SOURCE_USER_CONFIG = f"user global config file ({USER_CONFIG_PATH})"
SOURCE_LOCAL_CONFIG = f"local project config file ({LOCAL_CONFIG_PATH})"
SOURCE_ENV_VAR = "environment variable"
SOURCE_OVERRIDE = "runtime override"  # For values set via CLI options or future `config.set_runtime()`

# Environment variable prefixes
ENV_VAR_PREFIX = "COMPACT_MEMORY_"


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
                            if (
                                key in DEFAULT_CONFIG
                            ):  # Check against DEFAULT_CONFIG to ensure key is known
                                self._config[key] = value
                                self._sources[key] = SOURCE_USER_CONFIG
                            # else: print(f"Warning: Unknown key '{key}' in user config.")
            except Exception as e:
                print(f"Error loading user config: {e}", file=sys.stderr)

    def _load_local_config(self):
        if LOCAL_CONFIG_PATH.exists():
            try:
                with open(LOCAL_CONFIG_PATH, "r") as f:
                    local_config = yaml.safe_load(f)
                    if local_config and isinstance(local_config, dict):
                        for key, value in local_config.items():
                            if key in DEFAULT_CONFIG:  # Check against DEFAULT_CONFIG
                                self._config[key] = value
                                self._sources[key] = SOURCE_LOCAL_CONFIG
                        # else: print(f"Warning: Unknown key '{key}' in local config.")
            except Exception as e:
                print(f"Error loading local config: {e}", file=sys.stderr)

    def _load_env_vars(self):
        for key in DEFAULT_CONFIG.keys():
            env_var_name = ENV_VAR_PREFIX + key.upper()
            env_var_value_str = os.getenv(env_var_name)
            if env_var_value_str is not None:
                default_value = DEFAULT_CONFIG[key]
                try:
                    if isinstance(default_value, bool):
                        actual_value = env_var_value_str.lower() in ("true", "1", "yes")
                    elif isinstance(default_value, int):  # Added int casting
                        actual_value = int(env_var_value_str)
                    elif isinstance(default_value, float):  # Added float casting
                        actual_value = float(env_var_value_str)
                    else:
                        actual_value = env_var_value_str  # Assume string

                    self._config[key] = actual_value
                    self._sources[key] = f"{SOURCE_ENV_VAR} ({env_var_name})"
                except ValueError:
                    print(
                        f"Warning: Could not cast env var {env_var_name} value '{env_var_value_str}' to type {type(default_value)}. Using string value."
                    )
                    self._config[key] = env_var_value_str
                    self._sources[key] = f"{SOURCE_ENV_VAR} ({env_var_name})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any, source: str = SOURCE_OVERRIDE) -> bool:
        """
        Sets a configuration value in the runtime config and persists it to the user global config.
        Source can be specified, defaults to runtime override.
        Returns True if successful, False otherwise.
        """
        if key not in DEFAULT_CONFIG:
            print(
                f"Error: Configuration key '{key}' is not a recognized setting. Allowed keys are: {', '.join(DEFAULT_CONFIG.keys())}",
                file=sys.stderr,
            )
            # Or raise ValueError(f"Invalid configuration key: {key}")
            return False

        original_type = type(DEFAULT_CONFIG[key])

        # Attempt to cast the input string value to the original type
        if isinstance(value, str):  # All values from CLI Argument are strings initially
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
                print(
                    f"Error: Invalid value format for '{key}'. Cannot convert '{value}' to {original_type}.",
                    file=sys.stderr,
                )
                return False  # Indicate failure
        elif not isinstance(
            value, original_type
        ):  # If not a string and not the right type
            print(
                f"Error: Invalid type for '{key}'. Expected {original_type}, got {type(value)}.",
                file=sys.stderr,
            )
            return False  # Indicate failure
        # Type validation passed or value was already correct type

        self._config[key] = value
        self._sources[key] = source  # Runtime source

        # Persist to user global config file
        user_config_data = {}
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH, "r") as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        user_config_data = loaded_config
            except Exception as e:
                print(f"Error reading user config before set: {e}", file=sys.stderr)

        user_config_data[key] = value

        try:
            USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(USER_CONFIG_PATH, "w") as f:
                yaml.dump(user_config_data, f)
            return True  # Indicate success
        except Exception as e:
            print(f"Error writing to user config: {e}", file=sys.stderr)
            return False  # Indicate failure

    def get_with_source(self, key: str) -> Optional[Tuple[Any, str]]:
        if key in self._config:
            return self._config[key], self._sources.get(key, "Unknown")
        elif key in DEFAULT_CONFIG:  # Should be caught by _load_defaults
            return DEFAULT_CONFIG[key], SOURCE_DEFAULT
        return None

    def get_all_with_sources(self) -> Dict[str, Tuple[Any, str]]:
        all_data = {}
        # Start with all known default keys
        for key in DEFAULT_CONFIG.keys():
            all_data[key] = (
                self._config.get(key, DEFAULT_CONFIG[key]),
                self._sources.get(key, SOURCE_DEFAULT),
            )
        # Add any other keys that might have been set dynamically (e.g. from config files not in defaults)
        # This part is tricky if we strictly only want known keys.
        # For now, let's assume _config only contains keys that were either in DEFAULT_CONFIG or added via set()
        # and set() should ideally only allow known keys or handle them carefully.
        # The current implementation will show all items in self._config due to the loading logic.
        for key_present in self._config:
            if (
                key_present not in all_data
            ):  # Should not happen if loading logic is strict to DEFAULT_CONFIG
                all_data[key_present] = (
                    self._config[key_present],
                    self._sources.get(key_present, SOURCE_OVERRIDE),
                )
        return all_data

    def update_from_cli(self, key: str, value: Any):
        if value is not None:
            original_type = None
            if key in DEFAULT_CONFIG:
                original_type = type(DEFAULT_CONFIG[key])
                if not isinstance(value, original_type):  # CLI values are often strings
                    try:
                        if original_type is bool:
                            value = str(value).lower() in ("true", "1", "yes")
                        else:
                            value = original_type(value)
                    except ValueError:
                        print(
                            f"Warning: CLI value for '{key}' ('{value}') could not be cast to {original_type}. Using as is."
                        )

            self._config[key] = value
            self._sources[key] = "command-line argument"

    def validate(self) -> bool:  # Mostly for type validation after all loads
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in self._config:
                print(f"Validation Error: Missing configuration key: {key}")
                return False

            current_value = self._config[key]
            expected_type = type(default_value)

            if not isinstance(current_value, expected_type):
                try:
                    if expected_type is bool:
                        current_value = str(current_value).lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif expected_type is int:
                        current_value = int(str(current_value))
                    elif expected_type is float:
                        current_value = float(str(current_value))
                    else:
                        current_value = expected_type(
                            str(current_value)
                        )  # General attempt

                    if isinstance(current_value, expected_type):
                        self._config[key] = current_value
                    else:
                        raise ValueError("Coercion failed.")
                except ValueError:
                    print(
                        f"Validation Error: Key '{key}' has value '{self._config[key]}' of type {type(self._config[key])}, expected {expected_type}."
                    )
                    return False
        return True
