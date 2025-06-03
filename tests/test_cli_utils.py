import pytest
import json
from pathlib import Path
import typer # For typer.BadParameter

from gist_memory.cli_utils import parse_key_value_pairs, parse_complex_params

# Tests for parse_key_value_pairs
def test_pkvp_empty_and_none():
    assert parse_key_value_pairs(None) == {}
    assert parse_key_value_pairs([]) == {}

def test_pkvp_simple_strings():
    assert parse_key_value_pairs(["key1=value1", "key2=another value"]) == \
           {"key1": "value1", "key2": "another value"}

def test_pkvp_type_conversion():
    # json.loads will be used for values, so types are inferred
    assert parse_key_value_pairs(["int=123", "float=45.6", "bool_true=true", "bool_false=False", "null_val=null"]) == \
           {"int": 123, "float": 45.6, "bool_true": True, "bool_false": False, "null_val": None}

def test_pkvp_json_values():
    assert parse_key_value_pairs(['json_obj={"a":1,"b":"foo"}', 'json_list=[1,2,3]']) == \
           {"json_obj": {"a":1,"b":"foo"}, "json_list": [1,2,3]}
    # Test that strings that are not valid JSON primitives but are valid as part of a key=value pair remain strings
    assert parse_key_value_pairs(['quoted_str="hello world"']) == \
           {"quoted_str": "\"hello world\""} # json.loads will keep outer quotes for a string literal
    assert parse_key_value_pairs(['simple_str=hello world']) == \
           {"simple_str": "hello world"} # This is treated as a plain string

def test_pkvp_mixed_types_and_strings():
    # "007" will be a string because it's not a valid JSON number if not quoted,
    # and if it were "key=007" (no quotes), json.loads("007") is int 7.
    # If "key=\"007\"", then it's string "007".
    # The current implementation tries json.loads on the value part.
    assert parse_key_value_pairs(["path=/path/to/file", "num_str=007", "regular_str=stringval", "quoted_num_str=\"007\""]) == \
           {"path": "/path/to/file", "num_str": 7, "regular_str": "stringval", "quoted_num_str": "007"}

def test_pkvp_invalid_format():
    with pytest.raises(typer.BadParameter, match="Invalid parameter format 'key_no_value'. Expected 'key=value'."):
        parse_key_value_pairs(["key_no_value"])
    with pytest.raises(typer.BadParameter, match="Invalid parameter format ''. Expected 'key=value'."):
        parse_key_value_pairs([""]) # Empty string in list
    with pytest.raises(typer.BadParameter, match="Invalid parameter format ' =value_no_key'. Expected 'key=value'."):
        parse_key_value_pairs([" =value_no_key"])
    with pytest.raises(typer.BadParameter, match="Failed to decode JSON value for key 'bad_json'"):
        parse_key_value_pairs(["bad_json={not_json}"])


# Tests for parse_complex_params
@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "params.json"
    return file_path

def test_pcp_empty_and_none():
    assert parse_complex_params(None, None) == {}
    assert parse_complex_params(None, []) == {}
    assert parse_complex_params(params_file=None, params_kv=None) == {}


def test_pcp_from_file_only(temp_json_file: Path):
    data = {"file_key1": "file_value1", "file_int": 100}
    temp_json_file.write_text(json.dumps(data))
    assert parse_complex_params(params_file=temp_json_file, params_kv=None) == data

def test_pcp_from_kv_only():
    kv_data = ["kv_key1=kv_value1", "kv_bool=true"]
    expected = {"kv_key1": "kv_value1", "kv_bool": True}
    assert parse_complex_params(params_file=None, params_kv=kv_data) == expected

def test_pcp_file_and_kv_no_override(temp_json_file: Path):
    file_data = {"file_key": "file_val"}
    temp_json_file.write_text(json.dumps(file_data))
    kv_data = ["kv_key=kv_val"]
    expected = {"file_key": "file_val", "kv_key": "kv_val"}
    assert parse_complex_params(params_file=temp_json_file, params_kv=kv_data) == expected

def test_pcp_file_and_kv_with_override(temp_json_file: Path):
    file_data = {"common_key": "from_file", "file_only_key": "file_val"}
    temp_json_file.write_text(json.dumps(file_data))
    kv_data = ["common_key=\"from_kv\"", "kv_only_key=kv_val"] # Ensure "from_kv" is treated as a string
    expected = {"common_key": "from_kv", "file_only_key": "file_val", "kv_only_key": "kv_val"}
    assert parse_complex_params(params_file=temp_json_file, params_kv=kv_data) == expected

def test_pcp_file_not_found():
    non_existent_file = Path("nonexistent_params.json")
    with pytest.raises(typer.BadParameter, match=f"Parameters file not found: {non_existent_file}"):
        parse_complex_params(params_file=non_existent_file, params_kv=None)

def test_pcp_malformed_json_file(temp_json_file: Path):
    temp_json_file.write_text('{"key": "value", error_here}') # Malformed JSON
    with pytest.raises(typer.BadParameter, match=f"Error decoding JSON from '{temp_json_file}'"):
        parse_complex_params(params_file=temp_json_file, params_kv=None)

def test_pcp_json_file_not_dict(temp_json_file: Path):
    temp_json_file.write_text('[1, 2, 3]') # Valid JSON, but not a dictionary
    with pytest.raises(typer.BadParameter, match=f"Parameters file '{temp_json_file}' must contain a JSON object \\(dictionary\\)."):
        parse_complex_params(params_file=temp_json_file, params_kv=None)

def test_pcp_empty_json_file(temp_json_file: Path):
    temp_json_file.write_text('') # Empty file
    # json.loads('') raises JSONDecodeError, which is caught by parse_complex_params
    with pytest.raises(typer.BadParameter, match=f"Error decoding JSON from '{temp_json_file}'"):
        parse_complex_params(params_file=temp_json_file, params_kv=None)

def test_pcp_kv_override_types(temp_json_file: Path):
    file_data = {"num_val": "123", "bool_val": "true"} # Strings in file
    temp_json_file.write_text(json.dumps(file_data))
    # KV pairs should parse types correctly and override
    kv_data = ["num_val=456", "bool_val=false"]
    expected = {"num_val": 456, "bool_val": False}
    assert parse_complex_params(params_file=temp_json_file, params_kv=kv_data) == expected

def test_pkvp_unquoted_strings_vs_json_types():
    # Test how json.loads treats unquoted strings vs actual JSON types
    # This is mostly to document behavior of json.loads for simple values.
    assert parse_key_value_pairs(["str_val=mystring"]) == {"str_val": "mystring"}
    assert parse_key_value_pairs(["bool_val=true"]) == {"bool_val": True}
    assert parse_key_value_pairs(["int_val=10"]) == {"int_val": 10}
    assert parse_key_value_pairs(["float_val=10.5"]) == {"float_val": 10.5}
    assert parse_key_value_pairs(["null_val=null"]) == {"null_val": None}

    # If a string value happens to be "true", "false", "null", or a number, it'll be converted.
    # To keep it as a string, it must be JSON quoted in the value part.
    assert parse_key_value_pairs(["str_true=\"true\""]) == {"str_true": "true"}
    assert parse_key_value_pairs(["str_num=\"10\""]) == {"str_num": "10"}

def test_pkvp_equals_in_value():
    # Test if equals sign in value part is handled (it should be, as split('=', 1) is used)
    assert parse_key_value_pairs(["formula=e=mc^2"]) == {"formula": "e=mc^2"}
    assert parse_key_value_pairs(["json_val={\"equation\": \"e=mc^2\"}"]) == \
           {"json_val": {"equation": "e=mc^2"}}
