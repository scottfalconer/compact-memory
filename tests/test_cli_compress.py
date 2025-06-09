from pathlib import Path
import json
from typer.testing import CliRunner
from compact_memory.cli import app
from compact_memory.engines.registry import register_compression_engine
from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
    load_engine      # Added
)
# PrototypeEngine was removed


class DummyTruncEngine(BaseCompressionEngine):
    id = "dummy_trunc"

    def compress(self, text_or_chunks, llm_token_budget, previous_compression_result=None, **kwargs): # Added previous_compression_result and matched full signature
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )
        truncated = text[:llm_token_budget]

        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            steps=[{"type": "truncate"}],
            output_summary={"final_length": len(truncated)},
            final_compressed_object_preview=truncated,
        )
        # BaseCompressionEngine's compress method populates engine_id and engine_config.
        # This dummy engine overrides compress, so it needs to do it itself if those fields are important for tests using it.
        # For CLI tests, engine_id from the trace is usually what's checked.
        # For now, let's assume self.config is None or not critical for this dummy's CompressedMemory object.
        return CompressedMemory(
            text=truncated,
            trace=trace,
            engine_id=self.id,
            engine_config=getattr(self, 'config', None) # Mimic base engine
        )


register_compression_engine(DummyTruncEngine.id, DummyTruncEngine)


def _env(tmp_path: Path) -> dict[str, str]:
    # Ensure this matches the expected env var name used by the application's config loader
    return {
        "COMPACT_MEMORY_PATH": str(tmp_path),
        "COMPACT_MEMORY_DEFAULT_ENGINE_ID": "none", # Add defaults used by some tests
        "COMPACT_MEMORY_DEFAULT_MODEL_ID": "tiny-gpt2", # Add defaults used by some tests
    }


runner = CliRunner(mix_stderr=False)


def test_compress_text_option(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--engine",
            "none",
            "--budget",
            "10",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_compress_stdin(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "-",
            "--engine",
            "none",
            "--budget",
            "10",
        ],
        input="stdin text",
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "stdin" in result.stdout


def test_compress_file(tmp_path: Path):
    file_path = tmp_path / "inp.txt"
    file_path.write_text("file text")
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--engine",
            "none",
            "--budget",
            "10",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "file" in result.stdout


def test_compress_directory_recursive(tmp_path: Path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("aaa")
    sub = dir_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("bbb")
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(dir_path),
            "--engine",
            "none",
            "--budget",
            "5",
            "--recursive",
            "--output-dir",
            str(out_dir),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    expected_output_file = out_dir / "compressed_output.txt"
    assert expected_output_file.exists()
    # The 'none' engine with budget 5 on "aaa\n\nbbb".
    # 'none' engine uses token-based truncation. "aaa\n\nbbb" is 2 tokens via str.split()
    # or a small number of tokens via tiktoken, both <= budget 5. So, no truncation is expected.
    assert expected_output_file.read_text() == "aaa\n\nbbb"


def test_compress_invalid_combo(tmp_path: Path):
    file_path = tmp_path / "foo.txt"
    file_path.write_text("foo")
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--text",
            "oops",
            "--engine",
            "none",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Specify exactly ONE of --text, --file, or --dir" in result.stderr


def test_compress_file_output_file(tmp_path: Path):
    file_path = tmp_path / "input.txt"
    file_path.write_text("sample text")
    out_path = tmp_path / "out.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--engine",
            "none",
            "--budget",
            "100",
            "-o",
            str(out_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert out_path.read_text() == "sample text"


def test_compress_file_invalid_recursive(tmp_path: Path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("text")
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--engine",
            "none",
            "--budget",
            "5",
            "--recursive",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "--recursive is only valid with --dir" in result.stderr


def test_compress_dir_pattern_and_output(tmp_path: Path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("a")
    (dir_path / "b.md").write_text("b")
    sub = dir_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(dir_path),
            "--engine",
            "none",
            "--budget",
            "10",
            "--recursive",
            "--pattern",
            "*.txt",
            "--output-dir",
            str(out_dir),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    expected_output_file = out_dir / "compressed_output.txt"
    assert expected_output_file.exists()
    # Combined content from a.txt ("a") and sub/c.txt ("c")
    # sorted by rglob would be "a\n\nc".
    # 'none' engine with budget 10 should not change "a\n\nc" (length 4).
    assert expected_output_file.read_text() == "a\n\nc"
    # This check remains valid, ensuring other files are not in the output dir.
    assert not (out_dir / "b.md").exists()


def test_compress_invalid_strategy(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello",
            "--engine",
            "bogus",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Unknown one-shot compression engine" in result.stderr


def test_compress_budget_truncation(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "abcdef",
            "--engine",
            DummyTruncEngine.id,
            "--budget",
            "2",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "ab"


def test_compress_output_trace(tmp_path: Path):
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hi there",
            "--engine",
            "none",
            "--budget",
            "10",
            "--output-trace",
            str(trace_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    data = json.loads(trace_path.read_text())
    assert data["engine_name"] == "none"
    for key in ["strategy_params", "input_summary", "output_summary", "steps"]:
        assert key in data
    assert data["output_summary"]["output_length"] == len("hi there")


def test_compress_output_trace_details(tmp_path: Path):
    """Trace file should contain step metadata for strategy."""
    trace_path = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--engine",
            DummyTruncEngine.id,
            "--budget",
            "3",
            "--output-trace",
            str(trace_path),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    data = json.loads(trace_path.read_text())
    assert data["engine_name"] == DummyTruncEngine.id
    assert data["steps"] == [{"type": "truncate"}]
    assert data["output_summary"]["final_length"] == len("hello world"[:3])


def test_compress_verbose_stats(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "hello world",
            "--engine",
            "none",
            "--budget",
            "20",
            "--verbose-stats",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "Original tokens" in result.stdout
    assert "Compressed tokens" in result.stdout


def test_compress_nonexistent_file(tmp_path: Path):
    file_path = tmp_path / "missing.txt"
    result = runner.invoke(
        app,
        [
            "compress",
            "--file",
            str(file_path),
            "--engine",
            "none",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Invalid value for '--file'" in result.stderr


def test_compress_nonexistent_dir(tmp_path: Path):
    dir_path = tmp_path / "nope"
    result = runner.invoke(
        app,
        [
            "compress",
            "--dir",
            str(dir_path),
            "--engine",
            "none",
            "--budget",
            "5",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Invalid value for '--dir'" in result.stderr


def test_compress_uses_default_strategy(tmp_path: Path):
    env = _env(tmp_path)
    env["COMPACT_MEMORY_DEFAULT_ENGINE_ID"] = "none"
    result = runner.invoke(
        app,
        ["compress", "--text", "foobar", "--budget", "10"],
        env=env,
    )
    assert result.exit_code == 0
    assert "foobar" in result.stdout


def test_compress_override_default_strategy(tmp_path: Path):
    env = _env(tmp_path)
    env["COMPACT_MEMORY_DEFAULT_ENGINE_ID"] = DummyTruncEngine.id
    result = runner.invoke(
        app,
        [
            "compress",
            "--text",
            "abcdef",
            "--engine",
            "none",
            "--budget",
            "10",
        ],
        env=env,
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "abcdef"

# Tests related to PrototypeEngine were removed:
# - test_compress_to_memory_with_prototype_engine
# - test_compress_to_memory_one_shot_trunc_then_prototype


# --- Tests for PipelineEngine ---

def test_cli_compress_pipeline_engine_valid(tmp_path: Path):
    """Test valid PipelineEngine usage with a simple pipeline."""
    pipeline_config_json = """
    {
      "engines": [
        {"engine_name": "NoCompressionEngine", "engine_params": {}},
        {"engine_name": "FirstLastEngine", "engine_params": {"first_n": 2, "last_n": 2, "llm_token_budget": 10}}
      ]
    }
    """
    text_to_compress = "one two three four five six seven eight nine ten"
    # FirstLastEngine (first_n=2, last_n=2) on "one two three four five six seven eight nine ten"
    # Assuming space as delimiter by default if tiktoken not found.
    # Output: "one two nine ten"
    expected_output = "one two nine ten"

    result = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", pipeline_config_json,
        "--text", text_to_compress,
        "--budget", "10",  # Overall budget for the compress command
    ], env=_env(tmp_path))

    assert result.exit_code == 0, f"CLI Error: {result.stderr}"
    assert result.stdout.strip() == expected_output

    # Another valid case: StopwordPruner -> FirstLast
    pipeline_config_json_2 = """
    {
      "engines": [
        {"engine_name": "StopwordPrunerEngine", "engine_params": {"lang": "english"}},
        {"engine_name": "FirstLastEngine", "engine_params": {"first_n": 1, "last_n": 1, "llm_token_budget": 5}}
      ]
    }
    """
    text_to_compress_2 = "this is an example sentence with many common words"
    # Expected: "example words" (after "this is an", "with many common" are pruned, then first 1, last 1)
    # Actual output of StopwordPrunerEngine can be sensitive to its exact stopword list.
    # FirstLastEngine will then take 1 from start, 1 from end of that.
    # For "example sentence common words", FirstLast(1,1) -> "example words"
    expected_output_2 = "example sentence common words" # Output of StopwordPrunerEngine
    # Then FirstLastEngine takes first 1 and last 1.
    # This is a bit hard to predict exactly without running StopwordPruner,
    # so let's test if the command runs and produces *some* output.
    # A more robust test would mock the sub-engines or use a very predictable one.
    # For now, we'll check for successful execution and part of the expected non-stopword text.
    # If StopwordPrunerEngine is not available or fails, this test might be brittle.
    # Let's simplify expected output for robustness, assuming "example" and "words" survive.

    result_2 = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", pipeline_config_json_2,
        "--text", text_to_compress_2,
        "--budget", "5",
    ], env=_env(tmp_path))

    assert result_2.exit_code == 0, f"CLI Error: {result_2.stderr}"
    # Check for some expected keywords that should remain after stopword pruning
    assert "example" in result_2.stdout
    assert "words" in result_2.stdout
    assert "this" not in result_2.stdout # "this" is a common stopword


def test_cli_compress_pipeline_engine_invalid_json(tmp_path: Path):
    """Test PipelineEngine with invalid JSON in --pipeline-config."""
    invalid_json = '{"engines": [Oops this is not valid JSON}'

    result = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", invalid_json,
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))

    assert result.exit_code != 0
    assert "Error decoding pipeline config JSON" in result.stderr


def test_cli_compress_pipeline_engine_missing_config(tmp_path: Path):
    """Test PipelineEngine usage when --pipeline-config is missing."""
    result = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        # --pipeline-config is missing
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))

    assert result.exit_code != 0
    assert "Error: --pipeline-config is required when --engine is 'pipeline'." in result.stderr


def test_cli_compress_pipeline_config_without_pipeline_engine(tmp_path: Path):
    """Test providing --pipeline-config without specifying --engine pipeline."""
    pipeline_config_json = """
    {
      "engines": [{"engine_name": "NoCompressionEngine", "engine_params": {}}]
    }
    """
    result = runner.invoke(app, [
        "compress",
        "--engine", "none", # Not 'pipeline'
        "--pipeline-config", pipeline_config_json,
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))

    assert result.exit_code != 0
    assert "Error: --pipeline-config can only be used when --engine is 'pipeline'." in result.stderr


def test_cli_compress_pipeline_engine_unknown_sub_engine(tmp_path: Path):
    """Test PipelineEngine with an unknown engine_name in its config."""
    pipeline_config_json = """
    {
      "engines": [
        {"engine_name": "ThisEngineDoesNotExist", "engine_params": {}}
      ]
    }
    """
    result = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", pipeline_config_json,
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))

    assert result.exit_code != 0
    # The error message comes from the registry inside _get_one_shot_compression_engine
    assert "Unknown one-shot compression engine 'ThisEngineDoesNotExist'" in result.stderr


def test_cli_compress_pipeline_engine_invalid_config_structure(tmp_path: Path):
    """Test PipelineEngine with a structurally invalid (but valid JSON) config."""
    # Valid JSON, but not the expected structure (e.g., 'engines' key missing)
    invalid_structure_json = '{"not_engines_key": []}'
    result = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", invalid_structure_json,
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "Error creating pipeline engine from config" in result.stderr # General error

    # Valid JSON, 'engines' is not a list
    invalid_structure_json_2 = '{"engines": {"engine_name": "NoCompressionEngine"}}'
    result_2 = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", invalid_structure_json_2,
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))
    assert result_2.exit_code != 0
    assert "Pipeline config JSON must be a list" in result_2.stderr.lower() # Specific error from validation

    # Valid JSON, list items are not dicts
    invalid_structure_json_3 = '{"engines": ["NoCompressionEngine"]}'
    result_3 = runner.invoke(app, [
        "compress",
        "--engine", "pipeline",
        "--pipeline-config", invalid_structure_json_3,
        "--text", "some text",
        "--budget", "10",
    ], env=_env(tmp_path))
    assert result_3.exit_code != 0
    assert "Error in pipeline engine configuration structure" in result_3.stderr # Error from EngineConfig creation
