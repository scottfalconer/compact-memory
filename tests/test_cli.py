import json
from typer.testing import CliRunner
from gist_memory.cli import app
from gist_memory.embedding_pipeline import MockEncoder
import pytest


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )
    yield


def test_cli_dev_inspect_strategy(tmp_path):
    runner = CliRunner()
    # `agent init` takes path as direct argument
    result = runner.invoke(app, ["agent", "init", str(tmp_path), "--name", "tester"])
    assert result.exit_code == 0

    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("hello world")
    agent.store.save()

    # Pass global --memory-path before command group
    result = runner.invoke(
        app,
        [
            "--memory-path",  # Global option
            str(tmp_path),
            "dev",
            "inspect-strategy",
            "prototype",  # strategy_name argument
            # "--memory-path-arg", # This would be for specific override, not needed if global is set
            # str(tmp_path),
            "--list-prototypes",
        ],
    )
    assert result.exit_code == 0
    assert "hello world" in result.stdout


def test_cli_agent_stats(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["agent", "init", str(tmp_path), "--name", "tester"])
    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("alpha")
    agent.store.save()

    # Pass global --memory-path before command group, or use command-specific --memory-path-arg
    result = runner.invoke(
        app, ["--memory-path", str(tmp_path), "agent", "stats", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data["prototypes"] == 1


def test_cli_agent_validate_and_clear(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["agent", "init", str(tmp_path)])

    result = runner.invoke(app, ["--memory-path", str(tmp_path), "agent", "validate"])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()

    result = runner.invoke(
        app, ["--memory-path", str(tmp_path), "agent", "clear", "--force"]
    )  # --yes to --force
    assert result.exit_code == 0
    assert (
        not tmp_path.exists()
    )  # This check might be too strong if clear only empties dir


def test_cli_agent_validate_mismatch(tmp_path):
    runner = CliRunner()
    runner.invoke(app, ["agent", "init", str(tmp_path)])

    meta_path = tmp_path / "meta.yaml"  # meta.yaml is old, should be meta.json
    # Adjusting for meta.json as per current JsonNpyVectorStore
    meta_path_json = tmp_path / "meta.json"

    import json  # Using json instead of yaml for meta file

    meta = json.loads(meta_path_json.read_text())
    meta["embedding_dim"] = 2  # Intentionally make it mismatch
    meta_path_json.write_text(json.dumps(meta))

    result = runner.invoke(app, ["--memory-path", str(tmp_path), "agent", "validate"])
    assert result.exit_code != 0
    # The error message for mismatch might have changed, check if it's still a non-zero exit
    assert "mismatch" in result.stderr.lower()  # Checking stderr for error message


def test_cli_query(tmp_path, monkeypatch):  # Renamed from test_cli_talk
    runner = CliRunner()
    # Initialize agent first
    init_result = runner.invoke(app, ["agent", "init", str(tmp_path)])
    assert (
        init_result.exit_code == 0
    ), f"Agent init failed: {init_result.stdout + init_result.stderr}"

    from gist_memory.utils import load_agent

    agent = load_agent(tmp_path)
    agent.add_memory("hello world")
    agent.store.save()

    prompts = {}

    class DummyChatModel:  # Renamed class for clarity
        def __init__(self, *a, **kw):
            pass

        # Mock tokenizer if needed by the model, though not directly used by query's core logic shown
        # For this test, it's more about the interaction than perfect model simulation
        tokenizer = staticmethod(
            lambda text, return_tensors=None: {"input_ids": [text.split()]}
        )
        model = type(
            "M", (), {"config": type("C", (), {"n_positions": 50})()}
        )()  # Simplified mock
        max_new_tokens = 10

        def load_model(self):
            pass

        def prepare_prompt(
            self, agent_instance, prompt_text, **kw
        ):  # Matched expected signature if any
            return prompt_text

        def reply(self, prompt: str) -> str:
            prompts["text"] = prompt  # Capture the prompt for assertion
            return "mocked_response"

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", DummyChatModel)

    # Invoke query: global --memory-path, then command, then arguments
    result = runner.invoke(
        app,
        [
            "--memory-path",
            str(tmp_path),  # Global option
            "query",
            "hi?",  # query_text argument
        ],
    )
    assert (
        result.exit_code == 0
    ), f"Query command failed: {result.stdout + result.stderr}"
    assert "mocked_response" in result.stdout
    assert (
        "hello world" in prompts["text"]
    ), "Prompt given to LLM did not contain expected memory content"
    assert (
        "User asked: hi?" in prompts["text"]
    ), "Prompt given to LLM did not contain the user's query correctly formatted"


def test_query_command_calls_receive_channel_message(tmp_path, monkeypatch):  # Renamed
    runner = CliRunner()
    # Agent init needs to happen first to create the path structure if _load_agent expects it
    # However, we are mocking _load_agent, so a simple mkdir might be enough if _load_agent doesn't create files.
    # For safety and consistency with other tests, we can do a lightweight init or ensure path exists.
    # tmp_path.mkdir(exist_ok=True) # Ensuring path exists if init is not called.
    # Let's assume _load_agent is robust or we init an agent for this test too.
    init_result = runner.invoke(app, ["agent", "init", str(tmp_path)])  # Minimal init
    assert (
        init_result.exit_code == 0
    ), f"Agent init failed: {init_result.stdout + init_result.stderr}"

    called = {}

    class DummyAgent:
        store = type(
            "S", (), {"meta": {"default_model_id": "mock_model"}}
        )()  # Added default_model_id to meta

        # Ensure the signature matches what cli.query expects after loading an agent
        # and preparing for ActiveMemoryManager and compression strategy.
        def receive_channel_message(
            self, channel_id: str, text: str, mgr: Any, compression: Any = None
        ):
            called["channel_id"] = channel_id
            called["text"] = text
            called["mgr_type"] = type(mgr).__name__  # Store type name for assertion
            called["compression_provided"] = compression is not None
            return {"reply": "mocked_agent_reply"}

        # Mock _chat_model if its methods are called directly by query before receive_channel_message
        # Based on current cli.query, it seems LocalChatModel is instantiated separately
        # and agent._chat_model is assigned. So, this might not be needed on DummyAgent itself.

    monkeypatch.setattr("gist_memory.cli._load_agent", lambda p: DummyAgent())

    class DummyLLM:  # This is for the LocalChatModel instantiated in cli.query
        def __init__(self, *a, **kw):
            pass

        def load_model(self):
            pass

        # Add other methods if they are called, e.g. reply, prepare_prompt
        # but for this test, we are checking if agent.receive_channel_message is called.

    monkeypatch.setattr("gist_memory.local_llm.LocalChatModel", DummyLLM)

    result = runner.invoke(
        app,
        [
            "--memory-path",
            str(tmp_path),  # Global option
            "query",
            "hi?",  # query_text argument
        ],
    )
    assert (
        result.exit_code == 0
    ), f"Query command failed: {result.stdout + result.stderr}"
    assert called["channel_id"] == "cli"
    assert called["msg"] == "hi?"
    assert called["mgr_type"] == "ActiveMemoryManager"


def test_cli_download_chat_model(monkeypatch):
    runner = CliRunner()

    calls = []

    class DummyModelAutoClasses:  # Renamed for clarity
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append(name)
            return DummyModelAutoClasses()  # Return instance of itself

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        DummyModelAutoClasses.from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        DummyModelAutoClasses.from_pretrained,
    )

    result = runner.invoke(
        app, ["dev", "download-chat-model", "--model-name", "foo_model_test"]
    )  # Changed command path & model name
    assert (
        result.exit_code == 0
    ), f"dev download-chat-model failed: {result.stdout + result.stderr}"
    assert "foo_model_test" in calls


def test_cli_logging(tmp_path):
    log_path = tmp_path / "cli.log"
    runner = CliRunner()
    # agent init takes path as argument
    init_result = runner.invoke(app, ["agent", "init", str(tmp_path)])
    assert (
        init_result.exit_code == 0
    ), f"Agent init failed: {init_result.stdout + init_result.stderr}"

    from gist_memory.utils import load_agent  # Keep import local to where it's needed

    agent = load_agent(tmp_path)
    agent.add_memory("alpha_logging_test")  # Changed memory content for uniqueness
    agent.store.save()

    # Global options like --log-file, --verbose, and --memory-path should come before the command group
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_path),
            "--verbose",
            "--memory-path",
            str(tmp_path),  # Global memory path
            "agent",
            "stats",  # Command
            # No need for command-specific --memory-path-arg here as global is set
        ],
    )
    assert (
        result.exit_code == 0
    ), f"agent stats with logging failed: {result.stdout + result.stderr}"
    assert log_path.exists(), "Log file was not created"
    log_content = log_path.read_text()
    assert log_content != "", "Log file is empty"
    # A more robust check would be to look for specific log messages if possible
    assert (
        "alpha_logging_test" in log_content
        or "prototypes" in log_content
        or "DEBUG" in log_content
    )


def test_cli_corrupt_store_handling(tmp_path):  # Renamed for clarity
    runner = CliRunner()
    init_result = runner.invoke(app, ["agent", "init", str(tmp_path)])  # agent init
    assert (
        init_result.exit_code == 0
    ), f"Agent init failed: {init_result.stdout + init_result.stderr}"

    meta_path_json = tmp_path / "meta.json"  # Using meta.json

    import json  # Using json for meta file

    meta = json.loads(meta_path_json.read_text())
    meta["embedding_dim"] = 2  # Corrupting the embedding dimension
    meta_path_json.write_text(json.dumps(meta))

    # Invoke agent stats, expecting it to fail due to corruption
    result = runner.invoke(
        app,
        [
            "--memory-path",
            str(tmp_path),  # Global memory path
            "agent",
            "stats",  # Command
        ],
    )
    assert result.exit_code != 0, "Command should have failed due to corrupted store"
    # Check for a more generic corruption message if the specific one changed
    assert "corrupted" in result.stderr.lower() or "mismatch" in result.stderr.lower()


def test_cli_dev_inspect_trace(tmp_path):  # Renamed
    runner = CliRunner()
    trace_file = tmp_path / "trace.json"
    data = {
        "strategy_name": "dummy",
        "strategy_params": {},
        "input_summary": {},
        "steps": [{"type": "filter_item", "details": {"reason": "test"}}],
    }
    trace_file.write_text(json.dumps(data))
    result = runner.invoke(app, ["dev", "inspect-trace", str(trace_file)])
    assert (
        result.exit_code == 0
    ), f"dev inspect-trace failed: {result.stdout + result.stderr}"
    # Updated assertions based on the enhanced output of 'dev inspect-trace'
    assert "dummy" in result.stdout
    assert "filter_item" in result.stdout
    assert "reason" in result.stdout
    # Assuming the dev inspect-trace output format from cli.py shows these details from the trace data:
    # The cli.py dev inspect-trace shows: Strategy, idx, type, details preview.
    # The trace data includes: strategy_name, steps (with type, details).
    # It also has a title with strategy_name, original_tokens, compressed_tokens, processing_ms.
    # For this test, we are using a simple trace, let's ensure those fields are present if they are in the output.
    # The current cli.py code for inspect-trace shows:
    # console.print(f"Strategy: {data.get('strategy_name', '')}")
    # table.add_row(str(idx), stype or "", preview)
    # So, "Strategy: dummy" should be in the output.
    # And the table row will contain "filter_item" and a preview of {"reason": "test"}.
    assert "Strategy: dummy" in result.stdout  # Check strategy name is printed
    assert (
        "test" in result.stdout
    )  # Check if the reason "test" from details is in the output


def test_cli_compress_string_stdout():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compress",
            "alpha bravo charlie delta",
            "--strategy",
            "none",
            "--budget",
            "3",
        ],
    )
    assert (
        result.exit_code == 0
    ), f"compress string stdout failed: {result.stdout + result.stderr}"
    assert "alpha bravo charlie" in result.stdout


def test_cli_compress_string_to_file(tmp_path):
    runner = CliRunner()
    out_file = tmp_path / "out.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            "alpha bravo charlie delta",  # input_source argument
            "--strategy",
            "none",
            "--budget",
            "3",
            "-o",
            str(out_file),  # --output / -o option
        ],
    )
    assert (
        result.exit_code == 0
    ), f"compress string to file failed: {result.stdout + result.stderr}"
    assert out_file.read_text() == "alpha bravo charlie"


def test_cli_compress_file(tmp_path):
    runner = CliRunner()
    input_file = tmp_path / "in.txt"
    input_file.write_text("one two three four")
    out_file = tmp_path / "out.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            str(input_file),  # input_source argument
            "--strategy",
            "none",
            "--budget",
            "2",
            "-o",
            str(out_file),
        ],
    )
    assert (
        result.exit_code == 0
    ), f"compress file failed: {result.stdout + result.stderr}"
    assert out_file.read_text() == "one two"


def test_cli_compress_directory(tmp_path):
    runner = CliRunner()
    dir_in = tmp_path / "inputs"
    dir_in.mkdir()
    (dir_in / "a.txt").write_text("alpha bravo charlie")
    (dir_in / "b.txt").write_text("delta echo foxtrot")
    # For directory processing, compress command saves output files in the same directory
    # with _compressed suffix if -o is not a directory.
    result = runner.invoke(
        app,
        ["compress", str(dir_in), "--strategy", "none", "--budget", "2"],
    )
    assert (
        result.exit_code == 0
    ), f"compress directory failed: {result.stdout + result.stderr}"
    assert (dir_in / "a_compressed.txt").read_text() == "alpha bravo"
    assert (dir_in / "b_compressed.txt").read_text() == "delta echo"


def test_cli_compress_invalid_strategy():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["compress", "text", "--strategy", "bogus_strategy_name", "--budget", "3"],
    )
    assert (
        result.exit_code != 0
    ), f"compress with invalid strategy did not fail as expected: {result.stdout + result.stderr}"
    # Error message might have changed slightly, ensure it indicates strategy issue
    assert (
        "Unknown compression strategy" in result.stderr
        or "Invalid value for '--strategy'" in result.stderr
    )


def test_cli_compress_output_trace(tmp_path):
    runner = CliRunner()
    trace_file = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "compress",
            "hello world",  # input_source
            "--strategy",
            "none",
            "--budget",
            "5",
            "--output-trace",
            str(trace_file),
        ],
    )
    assert (
        result.exit_code == 0
    ), f"compress with --output-trace failed: {result.stdout + result.stderr}"
    assert trace_file.exists()
    trace_data = json.loads(trace_file.read_text())
    assert trace_data["strategy_name"] == "none"  # Check some trace content


def test_cli_dev_list_strategy_and_metric():  # Renamed
    runner = CliRunner()
    res = runner.invoke(
        app, ["dev", "list-strategies"]
    )  # strategy list -> dev list-strategies
    assert res.exit_code == 0, f"dev list-strategies failed: {res.stdout + res.stderr}"
    assert "none" in res.stdout  # 'none' strategy is usually available if 'default' is
    assert "prototype" in res.stdout  # Check for a common strategy

    res = runner.invoke(app, ["dev", "list-metrics"])  # metric list -> dev list-metrics
    assert res.exit_code == 0, f"dev list-metrics failed: {res.stdout + res.stderr}"
    assert "exact_match" in res.stdout


def test_cli_dev_evaluate_compression():  # Renamed
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",  # Added dev
            "abcde",
            "abcd",
            "--metric",
            "compression_ratio",
            "--json",
        ],
    )
    assert (
        res.exit_code == 0
    ), f"dev evaluate-compression failed: {res.stdout + res.stderr}"
    data = json.loads(res.stdout.strip())
    assert "compression_ratio" in data


def test_cli_dev_test_llm_prompt(monkeypatch, tmp_path):  # Renamed
    runner = CliRunner()

    class DummyProvider:
        def generate_response(
            self, prompt, model_name, max_new_tokens, api_key=None, **kw
        ):  # Added api_key
            DummyProvider.prompt = prompt
            return "ok"

        def get_token_budget(self, model_name, **kw):
            return 100

        def count_tokens(self, text, model_name, **kw):
            return len(text.split())

    monkeypatch.setattr(
        "gist_memory.llm_providers.OpenAIProvider",
        lambda: DummyProvider(),
    )
    monkeypatch.setattr(
        "gist_memory.llm_providers.GeminiProvider",
        lambda: DummyProvider(),
    )
    monkeypatch.setattr(
        "gist_memory.llm_providers.LocalTransformersProvider",
        lambda: DummyProvider(),
    )

    cfg_file = tmp_path / "llm_models_config.yaml"  # Matched default filename in cli.py
    cfg_file.write_text(
        "modelx:\n  provider: openai\n  model_name: modelx_openai_name"
    )  # Made model name more specific
    res = runner.invoke(
        app,
        [
            "dev",
            "test-llm-prompt",  # Added dev
            "--context",
            "foo_context",
            "--query",
            "bar_query",
            "--model",
            "modelx",  # This is the key in llm_models_config.yaml
            "--llm-config",
            str(cfg_file),
        ],
    )
    assert res.exit_code == 0, f"dev test-llm-prompt failed: {res.stdout + res.stderr}"
    assert "ok" in res.stdout
    assert "foo_context" in DummyProvider.prompt
    assert "bar_query" in DummyProvider.prompt


def test_cli_dev_evaluate_llm_response():  # Renamed
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "dev",
            "evaluate-llm-response",  # Added dev
            "foo_response",  # Made input more specific
            "bar_reference",  # Made input more specific
            "--metric",
            "exact_match",
            "--json",
        ],
    )
    assert (
        res.exit_code == 0
    ), f"dev evaluate-llm-response failed: {res.stdout + res.stderr}"
    data = json.loads(res.stdout.strip())
    assert "exact_match" in data


# --- Tests for 'config' command group ---

import os
import yaml
from pathlib import Path

# from gist_memory.config import DEFAULT_CONFIG # May not be needed directly in test_cli.py

# Note: PyYAML should be available as it's a dependency for gist_memory.config


def test_cli_config_show_defaults(tmp_path, monkeypatch):
    runner = CliRunner()
    # Ensure no user config file exists to see defaults
    # Path.home() is monkeypatched to return tmp_path
    # USER_CONFIG_PATH from config.py will use this mocked Path.home()

    # Construct the expected user config path based on the mocked home
    # This mirrors how Config class would find/create it.
    # USER_CONFIG_DIR = tmp_path / ".config" / "gist_memory" (from config.USER_CONFIG_DIR)
    # USER_CONFIG_PATH_EXPECTED = USER_CONFIG_DIR / "config.yaml" (from config.USER_CONFIG_PATH)

    # We need to use the actual USER_CONFIG_PATH from the SUT (System Under Test - config.py)
    # to ensure consistency if its definition ever changes slightly.
    # For this test, we ensure the file at the location Config class would use is deleted.

    from gist_memory.config import USER_CONFIG_PATH as SUT_USER_CONFIG_PATH

    # Monkeypatch Path.home() to make USER_CONFIG_PATH resolve within tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Resolve the SUT_USER_CONFIG_PATH with the monkeypatched home
    effective_user_config_file = Path(
        str(SUT_USER_CONFIG_PATH).replace("~", str(tmp_path))
    )

    if effective_user_config_file.exists():
        effective_user_config_file.unlink()
    # Ensure parent dirs exist for later 'set' commands if they were to create the file
    effective_user_config_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear relevant env vars
    monkeypatch.delenv("GIST_MEMORY_PATH", raising=False)
    monkeypatch.delenv("GIST_MEMORY_DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("GIST_MEMORY_DEFAULT_STRATEGY_ID", raising=False)
    # Assuming other potential env vars like GIST_MEMORY_VERBOSE, GIST_MEMORY_LOG_FILE are cleared by not setting them

    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, f"config show failed: {result.stdout}{result.stderr}"

    from gist_memory.config import DEFAULT_CONFIG  # Import here for direct comparison

    # The output of 'config show' is a table. Checking specific cell content is brittle.
    # Instead, check for key substrings and that "application default" is mentioned for each.
    for key, default_value in DEFAULT_CONFIG.items():
        assert key in result.stdout
        # For path, it's resolved, so check a unique part of the default path
        if key == "gist_memory_path":
            # Default path is like "~/.local/share/gist_memory". After expanduser, it's absolute.
            # With monkeypatched home, it's tmp_path + ".local/share/gist_memory"
            expected_path_segment = ".local/share/gist_memory"
            assert expected_path_segment in result.stdout
        else:
            assert str(default_value) in result.stdout
        # This is a bit weak as "application default" could be for another key.
        # A more robust way would be to parse the table or have JSON output for 'config show'.
        # For now, we check if "application default" appears generally for each key line.
        # This assumes each key's output is roughly on one line or closely grouped.

        # Find line containing the key
        key_line = ""
        for line in result.stdout.splitlines():
            if key in line:
                key_line = line
                break
        assert (
            "application default" in key_line
        ), f"Source for {key} not 'application default' or key not found in output line."


def test_cli_config_set_and_show_specific_key(tmp_path, monkeypatch):
    runner = CliRunner()

    from gist_memory.config import USER_CONFIG_PATH as SUT_USER_CONFIG_PATH

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    effective_user_config_file = Path(
        str(SUT_USER_CONFIG_PATH).replace("~", str(tmp_path))
    )

    if effective_user_config_file.exists():
        effective_user_config_file.unlink()
    effective_user_config_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear relevant env vars to ensure 'set' writes and 'show' reads from file
    monkeypatch.delenv("GIST_MEMORY_PATH", raising=False)
    monkeypatch.delenv("GIST_MEMORY_DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("GIST_MEMORY_VERBOSE", raising=False)  # For the verbose key test

    # Set a string value
    key_to_set = "default_model_id"
    value_to_set = "test_model_123_cli"
    result_set = runner.invoke(app, ["config", "set", key_to_set, value_to_set])
    assert (
        result_set.exit_code == 0
    ), f"config set failed for {key_to_set}: {result_set.stdout}{result_set.stderr}"
    assert f"Successfully set '{key_to_set}' to '{value_to_set}'" in result_set.stdout

    # Verify it's in the file
    assert effective_user_config_file.exists()
    with open(effective_user_config_file, "r") as f:
        content = yaml.safe_load(f)
    assert content[key_to_set] == value_to_set

    # Show the specific key
    result_show_key = runner.invoke(app, ["config", "show", "--key", key_to_set])
    assert result_show_key.exit_code == 0
    # Assuming table output, check for key, value, and source in the output string
    assert key_to_set in result_show_key.stdout
    assert value_to_set in result_show_key.stdout
    assert "user global config file" in result_show_key.stdout  # Source check

    # Set a boolean value (input as string "true")
    key_verbose = "verbose"  # This key exists in DEFAULT_CONFIG
    value_verbose_str = "true"
    expected_stored_bool = True

    result_set_verbose = runner.invoke(
        app, ["config", "set", key_verbose, value_verbose_str]
    )
    assert (
        result_set_verbose.exit_code == 0
    ), f"config set failed for {key_verbose}: {result_set_verbose.stdout}{result_set_verbose.stderr}"

    with open(effective_user_config_file, "r") as f:
        content_verbose = yaml.safe_load(f)
    assert (
        content_verbose[key_verbose] is expected_stored_bool
    )  # Should be stored as boolean by config.set

    result_show_verbose = runner.invoke(app, ["config", "show", "--key", key_verbose])
    assert result_show_verbose.exit_code == 0
    assert key_verbose in result_show_verbose.stdout
    assert (
        str(expected_stored_bool) in result_show_verbose.stdout
    )  # Output should reflect boolean
    assert "user global config file" in result_show_verbose.stdout


def test_cli_config_show_with_env_override(tmp_path, monkeypatch):
    runner = CliRunner()

    from gist_memory.config import (
        USER_CONFIG_PATH as SUT_USER_CONFIG_PATH,
        ENV_VAR_PREFIX,
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    effective_user_config_file = Path(
        str(SUT_USER_CONFIG_PATH).replace("~", str(tmp_path))
    )
    effective_user_config_file.parent.mkdir(parents=True, exist_ok=True)

    # Set a base value in the user config file
    user_setting_key = "gist_memory_path"
    user_setting_value = "/path/from/user_config_file_content"  # Made unique
    with open(effective_user_config_file, "w") as f:
        yaml.dump({user_setting_key: user_setting_value}, f)

    # Override with an environment variable
    env_key_name = f"{ENV_VAR_PREFIX}{user_setting_key.upper()}"
    env_value = "/path/from/env_var_test"
    monkeypatch.setenv(env_key_name, env_value)

    result = runner.invoke(app, ["config", "show", "--key", user_setting_key])
    assert (
        result.exit_code == 0
    ), f"config show with env override failed: {result.stdout}{result.stderr}"

    assert user_setting_key in result.stdout
    assert env_value in result.stdout  # Env value should be shown
    assert (
        f"environment variable ({env_key_name})" in result.stdout
    )  # Source should be env var

    # Clean up env var
    monkeypatch.delenv(env_key_name)


def test_cli_config_set_invalid_key(tmp_path, monkeypatch):
    runner = CliRunner()

    from gist_memory.config import USER_CONFIG_PATH as SUT_USER_CONFIG_PATH

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    effective_user_config_file = Path(
        str(SUT_USER_CONFIG_PATH).replace("~", str(tmp_path))
    )
    effective_user_config_file.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure dir exists

    result = runner.invoke(
        app, ["config", "set", "this_is_an_invalid_key", "some_value"]
    )
    assert result.exit_code != 0  # Should fail
    # Check stderr for the error message (cli.py prints error from config.set to typer.secho with err=True)
    assert (
        "Error: Configuration key 'this_is_an_invalid_key' is not a recognized setting."
        in result.stderr
    )


# --- Tests for interactive prompting ---
import sys
import typer  # Required for mocking typer.prompt


def test_cli_interactive_prompt_for_memory_path_if_tty_and_missing(
    tmp_path, monkeypatch
):
    runner = CliRunner()

    # Ensure no config files or env vars set memory_path
    # Path.home() is monkeypatched to return tmp_path for config file isolation
    from gist_memory.config import USER_CONFIG_PATH as SUT_USER_CONFIG_PATH

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    effective_user_config_file = Path(
        str(SUT_USER_CONFIG_PATH).replace("~", str(tmp_path))
    )
    if effective_user_config_file.exists():
        effective_user_config_file.unlink()
    effective_user_config_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure local .gmconfig.yaml doesn't exist in a potential CWD
    # For CLI runner tests, CWD is usually the project root or a temp dir created by runner.
    # To be safe, we can change CWD to tmp_path for this test.
    monkeypatch.chdir(tmp_path)
    local_gm_config = tmp_path / ".gmconfig.yaml"
    if local_gm_config.exists():
        local_gm_config.unlink()

    monkeypatch.delenv("GIST_MEMORY_PATH", raising=False)

    # Mock sys.stdin.isatty() to return True (interactive session)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Mock typer.prompt to provide input
    prompted_path = str(tmp_path / "prompted_agent_path_interactive")

    # Store if prompt was called with expected text
    prompt_info = {"called": False, "text_ok": False}

    def mock_typer_prompt(text, default=None, **kwargs):
        prompt_info["called"] = True
        if "Please enter the path for Gist Memory storage" in text:
            prompt_info["text_ok"] = True
        return prompted_path

    monkeypatch.setattr(typer, "prompt", mock_typer_prompt)

    # We need a command that would trigger the memory_path check.
    # 'agent stats' is a good candidate.
    result = runner.invoke(app, ["agent", "stats"])

    assert prompt_info["called"], "typer.prompt was not called"
    assert prompt_info["text_ok"], "typer.prompt was called with unexpected text"

    # Expected behavior:
    # 1. Prompt occurs, returns 'prompted_path'.
    # 2. 'agent stats' then tries to use 'prompted_path'.
    # 3. Since 'prompted_path' doesn't exist as an agent, 'agent stats' will show an error.
    assert (
        result.exit_code != 0
    )  # agent stats should fail on an empty/non-existent path
    # The error message from _load_agent in cli.py when path doesn't exist
    assert (
        f"Error: Memory path '{prompted_path}' not found or is invalid."
        in result.stderr
    )


def test_cli_no_interactive_prompt_if_not_tty_and_missing_path(tmp_path, monkeypatch):
    runner = CliRunner()

    from gist_memory.config import USER_CONFIG_PATH as SUT_USER_CONFIG_PATH

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    effective_user_config_file = Path(
        str(SUT_USER_CONFIG_PATH).replace("~", str(tmp_path))
    )
    if effective_user_config_file.exists():
        effective_user_config_file.unlink()
    effective_user_config_file.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)
    local_gm_config = tmp_path / ".gmconfig.yaml"
    if local_gm_config.exists():
        local_gm_config.unlink()

    monkeypatch.delenv("GIST_MEMORY_PATH", raising=False)

    # Mock sys.stdin.isatty() to return False (non-interactive session)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    prompt_called = False
    original_typer_prompt = typer.prompt  # Save original

    def spy_prompt(*args, **kwargs):
        nonlocal prompt_called
        prompt_called = True
        # This path should ideally not be taken in this test.
        # If it is, call the original to avoid breaking other unrelated prompts if any.
        return original_typer_prompt(*args, **kwargs)

    monkeypatch.setattr(typer, "prompt", spy_prompt)

    result = runner.invoke(
        app, ["agent", "stats"]
    )  # Use a command that requires memory_path

    assert result.exit_code != 0  # Should fail due to missing path
    assert not prompt_called  # Verify typer.prompt was NOT called
    assert (
        "Error: Gist Memory path is not set." in result.stderr
    )  # Specific non-interactive error
    assert "Please set it using the --memory-path option" in result.stderr

    monkeypatch.setattr(
        typer, "prompt", original_typer_prompt
    )  # Restore original prompt
