from pathlib import Path
from typer.testing import CliRunner
from compact_memory.cli import app

runner = CliRunner(env={"MIX_STDERR": "False"})


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "COMPACT_MEMORY_DEFAULT_ENGINE_ID": "none",
        "COMPACT_MEMORY_DEFAULT_MODEL_ID": "tiny-gpt2",
    }


def test_list_metrics(tmp_path: Path):
    result = runner.invoke(app, ["dev", "list-metrics"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "compression_ratio" in result.stdout
    assert "embedding_similarity" in result.stdout


def test_evaluate_compression_cli(tmp_path: Path, patch_embedding_model):
    result = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            "hello",
            "hello",
            "--metric",
            "embedding_similarity",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "similarity" in result.stdout


def test_evaluate_llm_response_cli(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "dev",
            "evaluate-llm-response",
            "hello",
            "hello",
            "--metric",
            "exact_match",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "exact_match" in result.stdout
