import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import typer


def parse_key_value_pairs(params: Optional[List[str]]) -> Dict[str, Any]:
    """
    Parses a list of strings like ["key1=value1", "key2=value2"] into a dictionary.
    Attempts to convert values to JSON types (numbers, booleans, null).
    """
    parsed_params: Dict[str, Any] = {}
    if not params:
        return parsed_params

    for p_str in params:
        if "=" not in p_str:
            raise typer.BadParameter(
                f"Invalid parameter format '{p_str}'. Expected 'key=value'."
            )

        key, value_str = p_str.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            raise typer.BadParameter(
                f"Invalid parameter format '{p_str}'. Expected 'key=value'."
            )

        parsed_value: Any

        if value_str.lower() in {"true", "false"}:
            parsed_value = value_str.lower() == "true"
        elif value_str.lower() == "null":
            parsed_value = None
        elif re.fullmatch(r"-?\d+", value_str):
            parsed_value = int(value_str)
        elif re.fullmatch(r"-?\d+\.\d*", value_str):
            parsed_value = float(value_str)
        elif value_str.startswith(("{", "[", '"', "'")):
            try:
                parsed_value = json.loads(value_str)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(
                    f"Failed to decode JSON value for key '{key}'"
                ) from e
        else:
            parsed_value = value_str

        parsed_params[key] = parsed_value

    return parsed_params


def parse_complex_params(
    params_file: Optional[Path] = None, params_kv: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Parses complex parameters from a JSON file and/or key-value pairs.
    Key-value pairs will override values from the JSON file if keys conflict.

    Args:
        params_file: Path to a JSON file.
        params_kv: List of key-value strings.

    Returns:
        A dictionary of the parsed parameters.
    """
    final_params: Dict[str, Any] = {}

    if params_file:
        if not params_file.exists():
            raise typer.BadParameter(f"Parameters file not found: {params_file}")
        try:
            content = params_file.read_text()
            final_params = json.loads(content)
            if not isinstance(final_params, dict):
                raise typer.BadParameter(
                    f"Parameters file '{params_file}' must contain a JSON object (dictionary)."
                )
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Error decoding JSON from '{params_file}'") from e
        except Exception as e:
            raise typer.BadParameter(
                f"Error reading parameters file {params_file}: {e}"
            )

    if params_kv:
        kv_dict = parse_key_value_pairs(params_kv)
        final_params.update(kv_dict)  # KV pairs override file params

    return final_params
