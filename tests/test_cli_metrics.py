import json
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from compact_memory.cli import app
from compact_memory.validation.embedding_metrics import (
    MultiModelEmbeddingSimilarityMetric,
)

# Import CompressionRatioMetric to mock its evaluate method if needed for evaluate-engines
from compact_memory.validation.compression_metrics import (
    CompressionRatioMetric,
)  # Corrected path

try:
    runner = CliRunner(mix_stderr=False)
except TypeError:
    runner = CliRunner()


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
    assert "multi_model_embedding_similarity" in result.stdout


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
    mock_get_tokenizer_direct_use = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.MultiModelEmbeddingSimilarityMetric._get_tokenizer",
        lambda s, m: mock_get_tokenizer_direct_use(m),
    )
    tok = MagicMock()
    tok.model_max_length = 8192
    mock_get_tokenizer_direct_use.return_value = tok
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


def test_evaluate_compression_multi_model_cli(tmp_path: Path, monkeypatch):
    original_text = "original sample text"
    compressed_text = "compressed sample text"
    metric_id = "multi_model_embedding_similarity"
    evaluate_method_path_on_class = "compact_memory.validation.embedding_metrics.MultiModelEmbeddingSimilarityMetric.evaluate"

    # Scenario 1
    mock_data_scen1 = {
        "embedding_similarity": {
            "mock-model-1": {"similarity": 0.8, "token_count": 10},
            "mock-model-2": {"similarity": 0.9, "token_count": 20},
        }
    }
    mock_evaluate_scen1 = MagicMock(return_value=mock_data_scen1)
    monkeypatch.setattr(evaluate_method_path_on_class, mock_evaluate_scen1)
    result_scen1 = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            original_text,
            compressed_text,
            "--metric",
            metric_id,
            "--embedding-model",
            "mock-model-1",
            "--embedding-model",
            "mock-model-2",
        ],
        env=_env(tmp_path),
    )
    assert (
        result_scen1.exit_code == 0
    ), f"STDOUT: {result_scen1.stdout}\nSTDERR: {result_scen1.stderr}"
    assert (
        "Scores for metric 'multi_model_embedding_similarity':" in result_scen1.stdout
    )
    assert "- mock-model-1: similarity 0.8, tokens 10" in result_scen1.stdout
    assert "- mock-model-2: similarity 0.9, tokens 20" in result_scen1.stdout
    mock_evaluate_scen1.assert_called_once()

    # Scenario 2
    mock_data_scen2 = {
        "embedding_similarity": {
            "mock-model-3": {"similarity": 0.7, "token_count": 15},
            "mock-model-4": {"similarity": 0.85, "token_count": 25},
        }
    }
    mock_evaluate_scen2 = MagicMock(return_value=mock_data_scen2)
    monkeypatch.setattr(evaluate_method_path_on_class, mock_evaluate_scen2)
    result_scen2 = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            original_text,
            compressed_text,
            "--metric",
            metric_id,
            "--embedding-model",
            '["mock-model-3", "mock-model-4"]',
        ],
        env=_env(tmp_path),
    )
    assert (
        result_scen2.exit_code == 0
    ), f"STDOUT: {result_scen2.stdout}\nSTDERR: {result_scen2.stderr}"
    assert (
        "Scores for metric 'multi_model_embedding_similarity':" in result_scen2.stdout
    )
    assert "- mock-model-3: similarity 0.7, tokens 15" in result_scen2.stdout
    assert "- mock-model-4: similarity 0.85, tokens 25" in result_scen2.stdout
    mock_evaluate_scen2.assert_called_once()

    # Scenario 3
    default_model_1 = "sentence-transformers/all-MiniLM-L6-v2"
    default_model_2 = "sentence-transformers/all-mpnet-base-v2"
    mock_data_scen3 = {
        "embedding_similarity": {
            default_model_1: {"similarity": 0.91, "token_count": 30},
            default_model_2: {"similarity": 0.92, "token_count": 35},
        }
    }
    mock_evaluate_scen3 = MagicMock(return_value=mock_data_scen3)
    monkeypatch.setattr(evaluate_method_path_on_class, mock_evaluate_scen3)
    result_scen3 = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            original_text,
            compressed_text,
            "--metric",
            metric_id,
        ],
        env=_env(tmp_path),
    )
    assert (
        result_scen3.exit_code == 0
    ), f"STDOUT: {result_scen3.stdout}\nSTDERR: {result_scen3.stderr}"
    assert (
        "Scores for metric 'multi_model_embedding_similarity':" in result_scen3.stdout
    )
    assert f"- {default_model_1}: similarity 0.91, tokens 30" in result_scen3.stdout
    assert f"- {default_model_2}: similarity 0.92, tokens 35" in result_scen3.stdout
    mock_evaluate_scen3.assert_called_once()

    # Scenario 4
    mock_data_scen4 = {
        "embedding_similarity": {
            "mock-model-1": {"similarity": 0.8, "token_count": 10},
            "mock-model-2": {"similarity": 0.9, "token_count": 20},
        }
    }
    mock_evaluate_scen4 = MagicMock(return_value=mock_data_scen4)
    monkeypatch.setattr(evaluate_method_path_on_class, mock_evaluate_scen4)
    result_scen4 = runner.invoke(
        app,
        [
            "dev",
            "evaluate-compression",
            original_text,
            compressed_text,
            "--metric",
            metric_id,
            "--embedding-model",
            "mock-model-1",
            "--embedding-model",
            "mock-model-2",
            "--json",
        ],
        env=_env(tmp_path),
    )
    assert (
        result_scen4.exit_code == 0
    ), f"STDOUT: {result_scen4.stdout}\nSTDERR: {result_scen4.stderr}"
    parsed_json_output = json.loads(result_scen4.stdout)
    assert parsed_json_output == mock_data_scen4
    mock_evaluate_scen4.assert_called_once()


def test_evaluate_engines_multi_model_cli(tmp_path: Path, monkeypatch):
    test_text = "This is example text for engine evaluation."
    engine_id = "none"
    # The evaluate-engines command runs a fixed set of metrics, including
    # MultiModelEmbeddingSimilarityMetric (results under "embedding_similarity" key)
    # and CompressionRatioMetric (results under "compression_ratio" key).

    # Mock for MultiModelEmbeddingSimilarityMetric.evaluate
    mms_evaluate_path = "compact_memory.validation.embedding_metrics.MultiModelEmbeddingSimilarityMetric.evaluate"
    mock_mms_evaluate_method = MagicMock()
    monkeypatch.setattr(mms_evaluate_path, mock_mms_evaluate_method)

    # Mock for CompressionRatioMetric.evaluate - to prevent it from actually calculating
    # and to control its output value.
    cr_evaluate_path = "compact_memory.validation.compression_metrics.CompressionRatioMetric.evaluate"  # Corrected path
    mock_cr_evaluate_method = MagicMock(return_value={"compression_ratio": 0.5})
    monkeypatch.setattr(cr_evaluate_path, mock_cr_evaluate_method)

    # Scenario 1 (was default run, now explicit): Multiple --embedding-model flags
    mock_mms_data_scen1 = {
        "embedding_similarity": {  # This is the structure MultiModelEmbeddingSimilarityMetric.evaluate returns
            "mock-model-1": {"similarity": 0.81, "token_count": 11},
            "mock-model-2": {"similarity": 0.91, "token_count": 21},
        }
    }
    mock_mms_evaluate_method.return_value = mock_mms_data_scen1

    result_scen1 = runner.invoke(
        app,
        [
            "dev",
            "evaluate-engines",
            "--text",
            test_text,
            "--engine",
            engine_id,
            # No --metrics flag, it runs default metrics including MultiModel...
            "--embedding-model",
            "mock-model-1",
            "--embedding-model",
            "mock-model-2",
        ],
        env=_env(tmp_path),
    )
    if result_scen1.exit_code != 0:
        print(f"Scenario 1 STDOUT: {result_scen1.stdout}")
        print(f"Scenario 1 STDERR: {result_scen1.stderr}")
    assert result_scen1.exit_code == 0
    mock_mms_evaluate_method.assert_called_once()
    mock_cr_evaluate_method.assert_called_once()  # Should also be called

    parsed_output_scen1 = json.loads(result_scen1.stdout)
    assert engine_id in parsed_output_scen1
    # The key in results is "embedding_similarity", not "multi_model_embedding_similarity"
    assert "embedding_similarity" in parsed_output_scen1[engine_id]
    assert (
        parsed_output_scen1[engine_id]["embedding_similarity"]
        == mock_mms_data_scen1["embedding_similarity"]
    )
    assert "compression_ratio" in parsed_output_scen1[engine_id]
    assert parsed_output_scen1[engine_id]["compression_ratio"] == 0.5

    mock_mms_evaluate_method.reset_mock()
    mock_cr_evaluate_method.reset_mock()

    # Scenario 2: JSON list for --embedding-model
    mock_mms_data_scen2 = {
        "embedding_similarity": {
            "mock-model-3": {"similarity": 0.71, "token_count": 16},
            "mock-model-4": {"similarity": 0.86, "token_count": 26},
        }
    }
    mock_mms_evaluate_method.return_value = mock_mms_data_scen2
    result_scen2 = runner.invoke(
        app,
        [
            "dev",
            "evaluate-engines",
            "--text",
            test_text,
            "--engine",
            engine_id,
            "--embedding-model",
            '["mock-model-3", "mock-model-4"]',
        ],
        env=_env(tmp_path),
    )
    assert (
        result_scen2.exit_code == 0
    ), f"STDOUT: {result_scen2.stdout}\nSTDERR: {result_scen2.stderr}"
    mock_mms_evaluate_method.assert_called_once()
    mock_cr_evaluate_method.assert_called_once()
    parsed_output_scen2 = json.loads(result_scen2.stdout)
    assert (
        parsed_output_scen2[engine_id]["embedding_similarity"]
        == mock_mms_data_scen2["embedding_similarity"]
    )
    assert parsed_output_scen2[engine_id]["compression_ratio"] == 0.5
    mock_mms_evaluate_method.reset_mock()
    mock_cr_evaluate_method.reset_mock()

    # Scenario 3: Default embedding models (no --embedding-model flag)
    default_model_1 = "sentence-transformers/all-MiniLM-L6-v2"
    default_model_2 = "sentence-transformers/all-mpnet-base-v2"
    mock_mms_data_default = {
        "embedding_similarity": {
            default_model_1: {"similarity": 0.93, "token_count": 31},
            default_model_2: {"similarity": 0.94, "token_count": 36},
        }
    }
    mock_mms_evaluate_method.return_value = mock_mms_data_default
    result_scen3 = runner.invoke(
        app,
        ["dev", "evaluate-engines", "--text", test_text, "--engine", engine_id],
        env=_env(tmp_path),
    )
    assert (
        result_scen3.exit_code == 0
    ), f"STDOUT: {result_scen3.stdout}\nSTDERR: {result_scen3.stderr}"
    mock_mms_evaluate_method.assert_called_once()
    mock_cr_evaluate_method.assert_called_once()
    parsed_output_scen3 = json.loads(result_scen3.stdout)
    assert (
        parsed_output_scen3[engine_id]["embedding_similarity"]
        == mock_mms_data_default["embedding_similarity"]
    )
    assert parsed_output_scen3[engine_id]["compression_ratio"] == 0.5

    # This part of the test (checking other metric not called) is not needed anymore
    # as evaluate-engines does not filter metrics via CLI option. It runs its predefined set.
