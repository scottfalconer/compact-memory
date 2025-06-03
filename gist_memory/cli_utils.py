import json
from pathlib import Path
from typing import List, Dict, Any, Optional
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
            # Allow flag-like parameters without values, defaulting them to True
            # This might be useful for boolean flags if not handled by Typer's own bool conversion.
            # However, for generic key-value, an explicit value is better.
            # For now, let's be strict. Revisit if use-case for valueless keys arises.
            raise typer.BadParameter(
                f"Invalid parameter format '{p_str}'. Expected 'key=value'."
            )
        key, value_str = p_str.split("=", 1)

        # Attempt to parse value as JSON (handles numbers, bools, null, actual JSON objects/arrays)
        try:
            parsed_value = json.loads(value_str)
        except json.JSONDecodeError:
            # If json.loads fails, it's not a valid JSON structure.
            # Handle specific string literals for common types like bool and None.
            if value_str.lower() == 'true':
                parsed_value = True
            elif value_str.lower() == 'false':
                parsed_value = False
            elif value_str.lower() == 'null':
                parsed_value = None
            else:
                # It's a plain string that isn't valid JSON (e.g., unquoted string)
                # or a string that happens to be "true", "false", "null" but intended as a string.
                # Typer/click usually handles direct string arguments well.
                # If strict JSON string is required, value_str should be enclosed in quotes.
                # Example: key="\"a string\""
                # For simple key=value like 'name=example', we treat 'example' as a string.
                parsed_value = value_str

        parsed_params[key] = parsed_value
    return parsed_params

def parse_complex_params(
    params_file: Optional[Path] = None,
    params_kv: Optional[List[str]] = None
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
            # It's better to raise FileNotFoundError directly or let the caller handle it.
            # However, for CLI context, BadParameter is appropriate.
            raise typer.BadParameter(f"Parameters file not found: {params_file}")
        try:
            content = params_file.read_text()
            if not content.strip(): # Handle empty file
                 raise typer.BadParameter(f"Parameters file '{params_file}' is empty.")
            final_params = json.loads(content)
            if not isinstance(final_params, dict):
                raise typer.BadParameter(
                    f"Parameters file '{params_file}' must contain a JSON object (dictionary)."
                )
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Error decoding JSON from {params_file}: {e}")
        except Exception as e: # Catch other potential errors like permission issues
            raise typer.BadParameter(f"Error reading parameters file {params_file}: {e}")

    if params_kv:
        kv_dict = parse_key_value_pairs(params_kv)
        final_params.update(kv_dict) # KV pairs override file params

    return final_params

if __name__ == '__main__':
    print("Testing parse_key_value_pairs:")
    # Test cases for parse_key_value_pairs
    test_kv_1 = ["a=1", "b=true", "c=null", "d=hello world", "e=\"quoted string\""]
    print(f"Input: {test_kv_1}, Output: {parse_key_value_pairs(test_kv_1)}")
    # Expected: {'a': 1, 'b': True, 'c': None, 'd': 'hello world', 'e': 'quoted string'}

    test_kv_2 = ["f={\"x\": 10, \"y\": \"test\"}", "g=[1, 2, \"three\"]"]
    print(f"Input: {test_kv_2}, Output: {parse_key_value_pairs(test_kv_2)}")
    # Expected: {'f': {'x': 10, 'y': 'test'}, 'g': [1, 2, 'three']}

    test_kv_3 = ["path=/path/to/file", "is_enabled=false", "count=0"]
    print(f"Input: {test_kv_3}, Output: {parse_key_value_pairs(test_kv_3)}")
    # Expected: {'path': '/path/to/file', 'is_enabled': False, 'count': 0}

    print(f"Input: None, Output: {parse_key_value_pairs(None)}")
    # Expected: {}

    try:
        parse_key_value_pairs(["invalid_format"])
    except typer.BadParameter as e:
        print(f"Caught expected error for invalid format: {e}")

    print("\nTesting parse_complex_params:")
    # Create dummy files for testing
    Path("test_params.json").write_text('{"file_key": "file_value", "common_key": "file_common", "file_bool": true, "file_num": 123.45}')
    Path("empty.json").write_text('')
    Path("malformed.json").write_text('{"file_key": "file_value", "common_key": "file_common",}') # trailing comma

    print(f"Test with file only: {parse_complex_params(params_file=Path('test_params.json'))}")
    # Expected: {'file_key': 'file_value', 'common_key': 'file_common', 'file_bool': True, 'file_num': 123.45}

    print(f"Test with KV only: {parse_complex_params(params_kv=['kv_key=kv_value', 'common_key=kv_common_override', 'kv_bool=true', 'kv_num=42'])}")
    # Expected: {'kv_key': 'kv_value', 'common_key': 'kv_common_override', 'kv_bool': True, 'kv_num': 42}

    print(f"Test with file and KV (KV overrides): {parse_complex_params(params_file=Path('test_params.json'), params_kv=['common_key=kv_override', 'another_kv=123', 'file_bool=false'])}")
    # Expected: {'file_key': 'file_value', 'common_key': 'kv_override', 'file_bool': False, 'file_num': 123.45, 'another_kv': 123}

    try:
        parse_complex_params(params_file=Path("nonexistent.json"))
    except typer.BadParameter as e:
        print(f"Caught expected error for nonexistent file: {e}")

    try:
        parse_complex_params(params_file=Path("empty.json"))
    except typer.BadParameter as e:
        print(f"Caught expected error for empty file: {e}")

    try:
        parse_complex_params(params_file=Path("malformed.json"))
    except typer.BadParameter as e:
        print(f"Caught expected error for malformed JSON: {e}")

    # Cleanup dummy files
    if Path("test_params.json").exists(): Path("test_params.json").unlink()
    if Path("empty.json").exists(): Path("empty.json").unlink()
    if Path("malformed.json").exists(): Path("malformed.json").unlink()
    print("Tests complete.")
