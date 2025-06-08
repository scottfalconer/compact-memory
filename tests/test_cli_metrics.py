from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import MagicMock
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
    assert "embedding_similarity_multi" in result.stdout


def test_evaluate_compression_cli(tmp_path: Path, patch_embedding_model):
    result = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            "hello",
            "hello",
            "--metric",
            "embedding_similarity_multi",
            "--metric-params",
            '{"model_names": ["model_a", "model_b"]}',
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0
    assert "semantic_similarity" in result.stdout
    assert "token_count" in result.stdout


def test_evaluate_compression_cli_openai_failure(tmp_path: Path, monkeypatch):
    openai_model = "openai/text-embedding-ada-002"

    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.ep.embed_text",
        lambda *a, **k: [[0.0, 0.0], [0.0, 0.0]],
    )
    mock_get_tokenizer = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.MultiModelEmbeddingSimilarityMetric._get_tokenizer",
        lambda *a, **k: mock_get_tokenizer(),
    )
    tok = MagicMock()
    tok.model_max_length = 8192
    mock_get_tokenizer.return_value = tok
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.token_utils.token_count",
        lambda *a, **k: 30,
    )

    result = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            "original text",
            "compressed text",
            "--metric",
            "multi_model_embedding_similarity",
            "--embedding-model",
            openai_model,
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert openai_model in result.stdout
    assert "tokens 30" in result.stdout


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
