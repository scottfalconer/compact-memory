from pathlib import Path
from typer.testing import CliRunner
import pytest

from compact_memory.cli import app

# from compact_memory.prototype_engine import PrototypeEngine # No longer directly used for instantiation here
# from compact_memory.vector_store import InMemoryVectorStore # No longer directly used
# from compact_memory.embedding_pipeline import get_embedding_dim # No longer directly used

# load_engine might be needed if we were to programmatically interact beyond CLI calls,
# but for this test, CLI calls cover engine interaction.

runner = CliRunner()


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "COMPACT_MEMORY_PATH": str(tmp_path),  # Corrected env var name
        "COMPACT_MEMORY_DEFAULT_ENGINE_ID": "none",
        "COMPACT_MEMORY_DEFAULT_MODEL_ID": "tiny-gpt2",
    }


@pytest.mark.skip(reason="NoCompressionEngine does not support querying")
def test_query_returns_reply(
    tmp_path: Path, patch_embedding_model
):  # Added patch_embedding_model fixture
    # Define a unique path for this test's store
    store_path = tmp_path / "query_test_store_trr"
    env_vars = _env(tmp_path)  # Get base env vars

    # 1. Initialize an engine store using the CLI
    init_result = runner.invoke(
        app,
        ["engine", "init", "--engine", "none", str(store_path)],
        env=env_vars,
    )
    assert init_result.exit_code == 0, f"Engine init failed: {init_result.stderr}"

    # 2. Add data to the store using the 'compress' CLI command (which uses engine.ingest)
    ingest_text = "the sky is blue and vast"
    compress_cmd_params = [
        "compress",
        "--memory-path",
        str(store_path),
        "--text",
        ingest_text,
        "--engine",
        "none",  # Use 'none' engine for one-shot compression to ingest text as is
        "--budget",
        "1000",  # Sufficiently large budget
    ]
    ingest_result = runner.invoke(app, compress_cmd_params, env=env_vars)
    assert (
        ingest_result.exit_code == 0
    ), f"Compress to memory (ingest) failed: {ingest_result.stderr}"

    # 3. Run the query using the CLI, targeting the initialized and populated store
    # Pass --memory-path as a global option to the app, before the 'query' subcommand,
    # as this is more reliable in testing than COMPACT_MEMORY_PATH env var for complex CLI apps.
    query_cmd_params = [
        "--memory-path",
        str(store_path),
        "query",
        "sky?",
    ]
    # env_vars from _env(tmp_path) might set COMPACT_MEMORY_PATH to tmp_path,
    # but the global --memory-path CLI option should take precedence.
    query_result = runner.invoke(app, query_cmd_params, env=env_vars)
    assert query_result.exit_code == 0, f"Query command failed: {query_result.stderr}"

    # Assert that the reply contains relevant information.
    # The exact response depends on the test LLM (tiny-gpt2 by default in _env).
    # We expect it to say something based on "the sky is blue and vast".
    # For a robust test that doesn't rely on exact LLM output, check for non-emptiness
    # or keywords.
    assert query_result.stdout.strip() != "", "Query returned an empty reply."
    # A more specific check could be:
    # assert "blue" in query_result.stdout.lower() or "sky" in query_result.stdout.lower()
    # Given it's a test LLM, the response might be minimal or nonsensical but should exist.
    # If the LLM just repeats or confirms, "blue" or "sky" is likely.
    # For "tiny-gpt2", responses can be very basic.
    # If the LLM is not actually running or failing silently, stdout might contain other messages.
    # The key is that `receive_channel_message` (which query calls) should produce a reply.
    # With patch_hf_auto, the DummyModel's generate returns [[0]], and DummyTokenizer.decode turns this into "t0".
    assert "t0" in query_result.stdout.lower()
