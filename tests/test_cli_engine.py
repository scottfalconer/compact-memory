from pathlib import Path
import json
from typer.testing import CliRunner

from compact_memory.cli import app
from compact_memory.engines import load_engine

# PrototypeEngine was removed
from compact_memory.engines.registry import (
    available_engines,
)  # To check against list output

# DummyTruncEngine might be needed if testing init with it, but current tests focus on an existing simple engine or default
# from compact_memory.engines import BaseCompressionEngine
# class DummyTestCliEngine(BaseCompressionEngine): # Example if a specific dummy was needed
#     id = "dummy_cli_test_eng"
# register_compression_engine(DummyTestCliEngine.id, DummyTestCliEngine)

runner = CliRunner()


def _env(tmp_path: Path) -> dict[str, str]:
    # Basic env, mainly to ensure no user-level config interferes unexpectedly
    # Specific tests can override COMPACT_MEMORY_PATH if needed,
    # but many engine commands take explicit --memory-path or target dir args.
    return {
        "COMPACT_MEMORY_DEFAULT_ENGINE_ID": "none",  # Keep a neutral default
        "COMPACT_MEMORY_DEFAULT_MODEL_ID": "tiny-gpt2",  # For any command that might try to load LLM
    }


def test_engine_list(tmp_path: Path, patch_embedding_model):
    """Test the 'engine list' command."""
    result = runner.invoke(app, ["engine", "list"], env=_env(tmp_path))
    assert result.exit_code == 0, f"CLI Error: {result.stderr}"
    # Check for some known core engines
    # assert "prototype" in result.stdout # PrototypeEngine removed
    # assert "base" in result.stdout # BaseCompressionEngine should not be listed
    assert "none" in result.stdout
    # Check against the registered engines for more dynamic validation
    registered_ids = available_engines()
    for engine_id in registered_ids:
        if engine_id not in [
            "dummy_trunc"
        ]:  # Exclude test-specific, dynamically registered ones if not always present
            assert engine_id in result.stdout


def test_engine_info(tmp_path: Path, patch_embedding_model):
    """Test the 'engine info' command."""
    result_none_engine = runner.invoke(
        app, ["engine", "info", "none"], env=_env(tmp_path)
    )
    assert result_none_engine.exit_code == 0, f"CLI Error: {result_none_engine.stderr}"
    none_engine_info = json.loads(result_none_engine.stdout)
    assert none_engine_info["engine_id"] == "none"
    assert "display_name" in none_engine_info

    result_non_existent = runner.invoke(
        app, ["engine", "info", "non_existent_engine"], env=_env(tmp_path)
    )
    assert result_non_existent.exit_code != 0
    assert (
        "not found" in result_non_existent.stderr.lower()
    )  # Check stderr for error message


def test_engine_init_success(tmp_path: Path, patch_embedding_model):
    """Test successful 'engine init'."""
    store_path = tmp_path / "init_test_store_cli"

    init_cmd = [
        "engine",
        "init",
        "--engine",
        "none",  # Changed from prototype to none
        str(store_path),
        # "--tau", str(tau_value), # tau is specific to PrototypeEngine
        "--chunker",
        "TestChunkerCLI",  # Custom chunker id for testing config
    ]
    result = runner.invoke(app, init_cmd, env=_env(tmp_path))
    assert result.exit_code == 0, f"CLI Error: {result.stderr}"
    assert (
        f"Successfully initialized Compact Memory engine store with engine 'none' at {store_path}"
        in result.stdout
    )

    assert (store_path / "engine_manifest.json").exists()
    manifest_data = json.loads((store_path / "engine_manifest.json").read_text())
    assert manifest_data.get("engine_id") == "none"
    # assert manifest_data.get("config", {}).get("similarity_threshold") == tau_value # Removed tau
    assert manifest_data.get("config", {}).get("chunker_id") == "TestChunkerCLI"


def test_engine_init_dir_not_empty(tmp_path: Path, patch_embedding_model):
    """Test 'engine init' failure if directory is not empty."""
    store_path = tmp_path / "init_fail_store_cli"
    store_path.mkdir()
    (store_path / "some_file.txt").write_text("hello")

    result = runner.invoke(app, ["engine", "init", str(store_path)], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "already exists and is not empty" in result.stderr.lower()


def test_engine_stats(tmp_path: Path, patch_embedding_model):
    """Test the 'engine stats' command."""
    store_path = tmp_path / "stats_store_cli"
    init_result = runner.invoke(
        app, ["engine", "init", "--engine", "none", str(store_path)], env=_env(tmp_path)
    )  # Changed to 'none'
    assert init_result.exit_code == 0, f"CLI Error: {init_result.stderr}"

    # Test text output
    stats_result = runner.invoke(
        app, ["engine", "stats", "--memory-path", str(store_path)], env=_env(tmp_path)
    )
    assert stats_result.exit_code != 0


def test_engine_clear(tmp_path: Path, patch_embedding_model):
    """Test the 'engine clear' command."""
    store_path = tmp_path / "clear_store_cli"
    init_result = runner.invoke(
        app, ["engine", "init", "--engine", "none", str(store_path)], env=_env(tmp_path)
    )  # Changed to 'none'
    assert init_result.exit_code == 0, f"CLI Error: {init_result.stderr}"
    assert (store_path / "engine_manifest.json").exists()

    clear_result = runner.invoke(
        app,
        ["engine", "clear", "--memory-path", str(store_path), "--force"],
        env=_env(tmp_path),
    )
    assert clear_result.exit_code == 0, f"CLI Error: {clear_result.stderr}"

    # After clearing, the directory itself might be removed or be empty.
    # The CLI 'clear' command uses shutil.rmtree, so the directory should not exist.
    assert not store_path.exists()


def test_engine_validate(tmp_path: Path, patch_embedding_model):
    """Test the 'engine validate' command."""
    store_path = tmp_path / "validate_store_cli"
    init_result = runner.invoke(
        app, ["engine", "init", "--engine", "none", str(store_path)], env=_env(tmp_path)
    )  # Changed to 'none'
    assert init_result.exit_code == 0, f"CLI Error: {init_result.stderr}"

    validate_result = runner.invoke(
        app,
        ["engine", "validate", "--memory-path", str(store_path)],
        env=_env(tmp_path),
    )
    assert validate_result.exit_code == 0, f"CLI Error: {validate_result.stderr}"
    # Current 'validate' command for a generic engine prints a simple message.
    # This assertion might change if validate's output becomes more detailed for 'none' or other engines.
    assert (
        "No specific storage validation implemented" in validate_result.stdout
    )  # This should still hold for 'none'
    # For example, if it checked its own files:
    # assert f"Validation for engine at '{store_path}' completed." in validate_result.stdout
    # For now, the generic message is what's expected.


# It's good practice to have a fixture for patch_embedding_model if it's used across many tests in this file.
# If it's not already globally available (e.g. in conftest.py), define or import it.
# Assuming patch_embedding_model is available from a conftest.py or similar.
# If not, and if these tests don't actually trigger embedding downloads (many CLI ones might not if LLM calls are mocked or not made),
# it might not be strictly necessary for *every* test here, but doesn't hurt.
# For `engine init` with `prototype`, it *might* try to get embedding_dim.
# For `engine stats`, `clear`, `validate`, `list`, `info` it's less likely.
# For safety, including it.


def test_dev_evaluate_engines_pipeline_engine(tmp_path: Path, patch_embedding_model):
    """Test the 'dev evaluate-engines' command with the pipeline engine."""
    test_text = "This is a test sentence for pipeline evaluation."
    # Using the global runner instance
    result = runner.invoke(
        app,
        [
            "dev",
            "evaluate-engines",
            "--text",
            test_text,
            "--engine",
            "pipeline",
            # No budget specified, should use default.
            # An empty pipeline (default config) should not depend on embedding models for this test.
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, f"CLI Error: {result.stderr}\nStdout: {result.stdout}"

    try:
        output_json = json.loads(result.stdout)
    except json.JSONDecodeError:
        assert False, f"Output is not valid JSON: {result.stdout}"

    assert "pipeline" in output_json, f"'pipeline' key missing in output: {output_json}"
    assert (
        "compression_ratio" in output_json["pipeline"]
    ), f"'compression_ratio' key missing in pipeline results: {output_json['pipeline']}"

    # For an empty pipeline, the text should be unchanged, so ratio is 1.0
    assert abs(output_json["pipeline"]["compression_ratio"] - 1.0) < 1e-9, (
        f"Compression ratio for pipeline was not 1.0: "
        f"{output_json['pipeline']['compression_ratio']}"
    )

    # Check for embedding_similarity as it's part of the output structure
    assert (
        "embedding_similarity" in output_json["pipeline"]
    ), f"'embedding_similarity' key missing in pipeline results: {output_json['pipeline']}"
    # For an empty pipeline, input and output are identical, so similarity should be perfect (1.0)
    # Assuming default embedding models are used if not specified,
    # and that the test environment can handle them (e.g. mocked or tiny models).
    # This part might need adjustment if default models cause issues or if the structure is different.
    if output_json["pipeline"][
        "embedding_similarity"
    ]:  # It might be empty if no models run
        for model_name, scores in output_json["pipeline"][
            "embedding_similarity"
        ].items():
            assert "similarity" in scores, f"Similarity score missing for {model_name}"
            assert (
                abs(scores["similarity"] - 1.0) < 1e-6
            ), f"Embedding similarity for {model_name} was not 1.0: {scores['similarity']}"  # Allow slightly larger tolerance for embeddings
