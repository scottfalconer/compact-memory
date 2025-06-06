import os
import pathlib
import sys
import yaml
from typing import Any, Dict, Optional, Tuple

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "compact_memory_path": "~/.local/share/compact_memory", # Default storage path, tilde will be expanded.
    "default_model_id": "openai/gpt-3.5-turbo", # Default model for LLM interactions.
    "default_engine_id": "none", # Default engine for operations like history compression. "none" (NoCompressionEngine) is a safe, valid default.
    "verbose": False, # Default verbosity.
    "log_file": None, # Default log file path (None means no file logging by default).
}

# Configuration file paths
USER_CONFIG_DIR = pathlib.Path("~/.config/compact_memory").expanduser()
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"
LOCAL_CONFIG_PATH = pathlib.Path(".gmconfig.yaml") # Project-level general config
LLM_MODELS_CONFIG_PATH = pathlib.Path("llm_models_config.yaml") # Project-level LLM models config

# Source descriptions
SOURCE_DEFAULT = "application default"
SOURCE_USER_CONFIG = f"user global config file ({USER_CONFIG_PATH})"
SOURCE_LOCAL_CONFIG = f"local project config file ({LOCAL_CONFIG_PATH})"
SOURCE_LLM_MODELS_CONFIG = f"llm models config file ({LLM_MODELS_CONFIG_PATH})"
SOURCE_ENV_VAR = "environment variable"
SOURCE_OVERRIDE = "runtime override"

# Environment variable prefixes
ENV_VAR_PREFIX = "COMPACT_MEMORY_"


class Config:
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}
        self.llm_configs: Dict[str, Any] = {} # For LLM model configurations

        self._load_defaults()
        self._load_user_config()
        self._load_local_config()
        self._load_env_vars()
        self._load_llm_models_config() # Load LLM specific configs
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
                            ):
                                self._config[key] = value
                                self._sources[key] = SOURCE_USER_CONFIG
            except Exception as e:
                print(f"Error loading user config '{USER_CONFIG_PATH}': {e}", file=sys.stderr)

    def _load_local_config(self):
        if LOCAL_CONFIG_PATH.exists():
            try:
                with open(LOCAL_CONFIG_PATH, "r") as f:
                    local_config = yaml.safe_load(f)
                    if local_config and isinstance(local_config, dict):
                        for key, value in local_config.items():
                            if key in DEFAULT_CONFIG:
                                self._config[key] = value
                                self._sources[key] = SOURCE_LOCAL_CONFIG
            except Exception as e:
                print(f"Error loading local config '{LOCAL_CONFIG_PATH}': {e}", file=sys.stderr)

    def _load_llm_models_config(self):
        """Loads LLM model configurations from LLM_MODELS_CONFIG_PATH."""
        if LLM_MODELS_CONFIG_PATH.exists():
            try:
                with open(LLM_MODELS_CONFIG_PATH, "r") as f:
                    loaded_llm_configs = yaml.safe_load(f)
                    if loaded_llm_configs and isinstance(loaded_llm_configs, dict):
                        self.llm_configs = loaded_llm_configs
                        # Optionally, log source for these configs if needed for debugging
                        # For simplicity, not adding to self._sources for each llm_config key
                    elif loaded_llm_configs is not None: # File exists but is not a dict (e.g. empty or just a list/value)
                         print(f"Warning: LLM models config file '{LLM_MODELS_CONFIG_PATH}' does not contain a valid dictionary.", file=sys.stderr)
            except yaml.YAMLError as e:
                print(f"Error parsing LLM models config file '{LLM_MODELS_CONFIG_PATH}': {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error loading LLM models config file '{LLM_MODELS_CONFIG_PATH}': {e}", file=sys.stderr)
        # else:
            # print(f"Info: LLM models config file '{LLM_MODELS_CONFIG_PATH}' not found. No LLM specific configs loaded.", file=sys.stderr)


    def _load_env_vars(self):
        for key in DEFAULT_CONFIG.keys():
            env_var_name = ENV_VAR_PREFIX + key.upper()
            env_var_value_str = os.getenv(env_var_name)
            if env_var_value_str is not None:
                default_value = DEFAULT_CONFIG[key]
                try:
                    if isinstance(default_value, bool):
                        actual_value = env_var_value_str.lower() in ("true", "1", "yes")
                    elif isinstance(default_value, int):
                        actual_value = int(env_var_value_str)
                    elif isinstance(default_value, float):
                        actual_value = float(env_var_value_str)
                    else:
                        if key == "compact_memory_path" and env_var_value_str:
                            try:
                                actual_value = str(pathlib.Path(env_var_value_str).expanduser().resolve())
                            except Exception as e:
                                print(f"Warning: Could not expand/resolve path '{env_var_value_str}' for {env_var_name}. Using raw value. Error: {e}", file=sys.stderr)
                                actual_value = env_var_value_str
                        else:
                            actual_value = env_var_value_str

                    self._config[key] = actual_value
                    self._sources[key] = f"{SOURCE_ENV_VAR} ({env_var_name})"
                except ValueError:
                    print(
                        f"Warning: Could not cast env var {env_var_name} value '{env_var_value_str}' to type {type(default_value)}. Using string value.", file=sys.stderr
                    )
                    if key == "compact_memory_path" and isinstance(env_var_value_str, str) and env_var_value_str:
                        try:
                            self._config[key] = str(pathlib.Path(env_var_value_str).expanduser().resolve())
                        except Exception as e:
                            print(f"Warning: Could not expand path '{env_var_value_str}' for {env_var_name} after casting error. Using raw value. Error: {e}", file=sys.stderr)
                            self._config[key] = env_var_value_str
                    else:
                        self._config[key] = env_var_value_str
                    self._sources[key] = f"{SOURCE_ENV_VAR} ({env_var_name})"


    def get(self, key: str, default: Any = None) -> Any:
        value = self._config.get(key, default)
        if key == "compact_memory_path" and isinstance(value, str):
            source = self._sources.get(key)
            if source and source != SOURCE_ENV_VAR and source != SOURCE_OVERRIDE:
                if "~" in value or (not pathlib.Path(value).is_absolute() and not value.startswith(os.path.sep + os.path.sep)):
                    try:
                        return str(pathlib.Path(value).expanduser().resolve())
                    except Exception as e:
                        print(f"Warning: Could not expand/resolve path '{value}' for {key} from {source}. Using original value. Error: {e}", file=sys.stderr)
                        return value
        return value

    def set(self, key: str, value: Any, source: str = SOURCE_OVERRIDE) -> bool:
        if key not in DEFAULT_CONFIG:
            print(
                f"Error: Configuration key '{key}' is not a recognized setting. Allowed keys are: {', '.join(DEFAULT_CONFIG.keys())}",
                file=sys.stderr,
            )
            return False

        original_type = type(DEFAULT_CONFIG[key])

        if isinstance(value, str):
            try:
                if original_type is bool:
                    value = value.lower() in ("true", "1", "yes", "on")
                elif original_type is int:
                    value = int(value)
                elif original_type is float:
                    value = float(value)
            except ValueError:
                print(
                    f"Error: Invalid value format for '{key}'. Cannot convert '{value}' to {original_type}.",
                    file=sys.stderr,
                )
                return False
        elif not isinstance(value, original_type):
            print(
                f"Error: Invalid type for '{key}'. Expected {original_type}, got {type(value)}.",
                file=sys.stderr,
            )
            return False

        self._config[key] = value
        self._sources[key] = source

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
            return True
        except Exception as e:
            print(f"Error writing to user config: {e}", file=sys.stderr)
            return False

    def get_with_source(self, key: str) -> Optional[Tuple[Any, str]]:
        if key in self._config:
            return self._config[key], self._sources.get(key, "Unknown")
        elif key in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[key], SOURCE_DEFAULT
        return None

    def get_all_with_sources(self) -> Dict[str, Tuple[Any, str]]:
        all_data = {}
        for key in DEFAULT_CONFIG.keys():
            all_data[key] = (
                self._config.get(key, DEFAULT_CONFIG[key]),
                self._sources.get(key, SOURCE_DEFAULT),
            )
        for key_present in self._config:
            if key_present not in all_data:
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
                if not isinstance(value, original_type):
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

    def validate(self) -> bool:
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
                        )

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

    # --- LLM Model Config Methods ---
    def get_llm_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Returns a specific LLM configuration by its name.
        Name usually corresponds to a model identifier like 'gpt-4-turbo' or 'local_default_summarizer'.
        """
        return self.llm_configs.get(name)

    def get_all_llm_configs(self) -> Dict[str, Any]:
        """
        Returns the entire dictionary of loaded LLM configurations.
        """
        return self.llm_configs
