import numpy as np
from unittest.mock import MagicMock, patch
import pytest
import tiktoken
import transformers

from compact_memory.validation.embedding_metrics import (
    MultiModelEmbeddingSimilarityMetric,
)
from compact_memory import embedding_pipeline as ep


def test_multi_model_embedding_similarity_basic(monkeypatch):
    mock_autotokenizer_class = MagicMock(spec=transformers.AutoTokenizer)
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.model_max_length = 128
    mock_autotokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
    monkeypatch.setattr(transformers, "AutoTokenizer", mock_autotokenizer_class)

    mock_tu_token_count = MagicMock(side_effect=[1, 1, 1])
    monkeypatch.setattr('compact_memory.validation.embedding_metrics.token_utils.token_count', mock_tu_token_count)

    fixed_vector = [1.0, 0.0]
    mock_ep_embed = MagicMock(return_value=[fixed_vector, fixed_vector])
    monkeypatch.setattr(ep, "embed_text", mock_ep_embed)

    metric = MultiModelEmbeddingSimilarityMetric(model_names=["dummy-model"])
    scores = metric.evaluate(original_text="hello", compressed_text="hello")

    assert "dummy-model" in scores["embedding_similarity"]
    data = scores["embedding_similarity"]["dummy-model"]
    assert np.isclose(data["similarity"], 1.0)
    assert data["token_count"] == 1


def test_multi_model_embedding_similarity_skip_long(monkeypatch, caplog):
    mock_autotokenizer_class = MagicMock(spec=transformers.AutoTokenizer)
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.model_max_length = 128
    mock_autotokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
    monkeypatch.setattr(transformers, "AutoTokenizer", mock_autotokenizer_class)

    long_text_token_count_a = 200
    long_text_token_count_b = 200
    mock_tu_token_count = MagicMock(side_effect=[long_text_token_count_a, long_text_token_count_b])
    monkeypatch.setattr('compact_memory.validation.embedding_metrics.token_utils.token_count', mock_tu_token_count)

    mock_ep_embed = MagicMock()
    monkeypatch.setattr(ep, "embed_text", mock_ep_embed)

    metric = MultiModelEmbeddingSimilarityMetric(model_names=["dummy-model-hf"])
    long_text = " ".join("t" + str(i) for i in range(200))
    scores = metric.evaluate(original_text=long_text, compressed_text=long_text)

    assert scores["embedding_similarity"] == {}
    mock_ep_embed.assert_not_called()
    assert any(record.levelname == "WARNING" and "Input exceeds model_max_length for dummy-model-hf; skipping" in record.message for record in caplog.records)


def test_multi_model_embedding_similarity_multiple_hf_models(monkeypatch, caplog):
    mock_autotokenizer_class = MagicMock(spec=transformers.AutoTokenizer)
    mock_hf_tokenizer_instance_1 = MagicMock(); mock_hf_tokenizer_instance_1.model_max_length = 128
    mock_hf_tokenizer_instance_2 = MagicMock(); mock_hf_tokenizer_instance_2.model_max_length = 128

    def from_pretrained_side_effect(model_name_arg, **kwargs):
        if model_name_arg == "hf_model_1": return mock_hf_tokenizer_instance_1
        if model_name_arg == "hf_model_2": return mock_hf_tokenizer_instance_2
        pytest.fail(f"Unexpected model name for AutoTokenizer: {model_name_arg}")
    mock_autotokenizer_class.from_pretrained.side_effect = from_pretrained_side_effect
    monkeypatch.setattr(transformers, "AutoTokenizer", mock_autotokenizer_class)

    mock_embed_text_hf = MagicMock()
    monkeypatch.setattr(ep, "embed_text", mock_embed_text_hf)

    mock_token_count = MagicMock(side_effect=[ 10, 10, 10, 20, 20, 20 ])
    monkeypatch.setattr('compact_memory.validation.embedding_metrics.token_utils.token_count', mock_token_count)

    def embed_text_side_effect_multi(texts, model_name, **kwargs):
        if model_name == "hf_model_1": return [[0.1, 0.2], [0.3, 0.4]]
        if model_name == "hf_model_2": return [[0.5, 0.6], [0.7, 0.8]]
        pytest.fail(f"Unexpected model name for embed_text: {model_name}")
    mock_embed_text_hf.side_effect = embed_text_side_effect_multi

    metric = MultiModelEmbeddingSimilarityMetric(model_names=["hf_model_1", "hf_model_2"])
    result = metric.evaluate(original_text="original text", compressed_text="compressed text")

    assert "embedding_similarity" in result; es_results = result["embedding_similarity"]
    assert "hf_model_1" in es_results
    assert np.isclose(es_results["hf_model_1"]["similarity"], np.dot([0.1, 0.2], [0.3, 0.4]))
    assert es_results["hf_model_1"]["token_count"] == 10
    assert "hf_model_2" in es_results
    assert np.isclose(es_results["hf_model_2"]["similarity"], np.dot([0.5, 0.6], [0.7, 0.8]))
    assert es_results["hf_model_2"]["token_count"] == 20


def test_multi_model_openai_embedding_failure(monkeypatch, caplog):
    mock_ep_embed_text = MagicMock()
    monkeypatch.setattr(ep, "embed_text", mock_ep_embed_text)
    mock_token_utils_token_count = MagicMock()
    monkeypatch.setattr('compact_memory.validation.embedding_metrics.token_utils.token_count', mock_token_utils_token_count)

    mock_actual_get_tokenizer = MagicMock()
    mock_openai_tokenizer_inst = MagicMock(spec=tiktoken.Encoding)
    mock_openai_tokenizer_inst.model_max_length = 8192
    mock_actual_get_tokenizer.return_value = mock_openai_tokenizer_inst
    monkeypatch.setattr(MultiModelEmbeddingSimilarityMetric, "_get_tokenizer", mock_actual_get_tokenizer)

    openai_model_name = "openai/text-embedding-ada-002"
    mock_ep_embed_text.side_effect = Exception("Simulated embedding error")
    mock_token_utils_token_count.side_effect = [30,30,30]

    metric = MultiModelEmbeddingSimilarityMetric(model_names=[openai_model_name])
    result = metric.evaluate(original_text="original text", compressed_text="compressed text")

    mock_actual_get_tokenizer.assert_called_with(openai_model_name)
    assert "embedding_similarity" in result; es_results = result["embedding_similarity"]
    assert openai_model_name in es_results; openai_result = es_results[openai_model_name]
    assert openai_result["token_count"] == 30
    assert np.isclose(openai_result["similarity"], 0.0)
    assert any(record.levelname == "WARNING" and f"Embedding failed for {openai_model_name}" in record.message for record in caplog.records)


@patch('tiktoken.get_encoding', autospec=True)
@patch('tiktoken.encoding_for_model', autospec=True)
def test_openai_tokenizer_max_length_determination(mock_encoding_for_model, mock_get_encoding):
    metric = MultiModelEmbeddingSimilarityMetric()
    model_name_ada = "openai/text-embedding-ada-002"; expected_max_len_ada = 8191
    mock_tok_ada = MagicMock(spec=tiktoken.Encoding); mock_encoding_for_model.return_value = mock_tok_ada
    tokenizer_ada = metric._get_tokenizer(model_name_ada)
    assert tokenizer_ada.model_max_length == expected_max_len_ada; mock_encoding_for_model.assert_called_with(model_name_ada.split("/", 1)[1])
    mock_encoding_for_model.reset_mock(); mock_get_encoding.reset_mock()
    model_name_custom_mml = "openai/custom-mml-model"; expected_max_len_custom_mml = 4000
    mock_tok_custom_mml = MagicMock(spec=tiktoken.Encoding); mock_tok_custom_mml.model_max_length = expected_max_len_custom_mml; mock_tok_custom_mml.n_ctx = None
    mock_encoding_for_model.return_value = mock_tok_custom_mml
    tokenizer_custom_mml = metric._get_tokenizer(model_name_custom_mml)
    assert tokenizer_custom_mml.model_max_length == expected_max_len_custom_mml
    mock_encoding_for_model.reset_mock(); mock_get_encoding.reset_mock()
    model_name_nctx = "openai/custom-nctx-model"; expected_max_len_nctx = 2048
    mock_tok_nctx = MagicMock(spec=tiktoken.Encoding); mock_tok_nctx.model_max_length = None; mock_tok_nctx.n_ctx = expected_max_len_nctx
    mock_encoding_for_model.return_value = mock_tok_nctx
    tokenizer_nctx = metric._get_tokenizer(model_name_nctx)
    assert tokenizer_nctx.model_max_length == expected_max_len_nctx
    mock_encoding_for_model.reset_mock(); mock_get_encoding.reset_mock()
    model_name_unknown = "openai/unknown-model"; expected_max_len_default = 8191
    mock_tok_unknown = MagicMock(spec=tiktoken.Encoding); mock_tok_unknown.model_max_length = None; mock_tok_unknown.n_ctx = None
    mock_encoding_for_model.return_value = mock_tok_unknown
    tokenizer_unknown = metric._get_tokenizer(model_name_unknown)
    assert tokenizer_unknown.model_max_length == expected_max_len_default
    mock_encoding_for_model.reset_mock(); mock_get_encoding.reset_mock()
    model_name_unavailable = "openai/unavailable-model"; expected_max_len_gpt2_fallback = 1024
    mock_encoding_for_model.side_effect = Exception("Test encoding failure")
    mock_gpt2_tokenizer = MagicMock(spec=tiktoken.Encoding); mock_gpt2_tokenizer.model_max_length = None; mock_gpt2_tokenizer.n_ctx = expected_max_len_gpt2_fallback
    mock_get_encoding.return_value = mock_gpt2_tokenizer
    tokenizer_fallback = metric._get_tokenizer(model_name_unavailable)
    assert tokenizer_fallback.model_max_length == expected_max_len_gpt2_fallback; mock_get_encoding.assert_called_once_with("gpt2")


def test_openai_token_counting_and_skipping(monkeypatch, caplog):
    openai_model_name = "openai/text-embedding-ada-002"; ada_base_model_name = "text-embedding-ada-002"
    metric = MultiModelEmbeddingSimilarityMetric(model_names=[openai_model_name])
    original_text = "This is a short original text."; compressed_text = "This is a short compressed text."
    mock_embed_text_oai = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(ep, "embed_text", mock_embed_text_oai)
    results = metric.evaluate(original_text=original_text, compressed_text=compressed_text)
    try: enc = tiktoken.encoding_for_model(ada_base_model_name); expected_token_count = len(enc.encode(compressed_text))
    except Exception as e: pytest.fail(f"Failed to get tiktoken encoding for {ada_base_model_name}: {e}")
    assert openai_model_name in results["embedding_similarity"]
    model_results = results["embedding_similarity"][openai_model_name]
    assert model_results["token_count"] == expected_token_count
    assert np.isclose(model_results["similarity"], np.dot([0.1, 0.2], [0.3, 0.4]))
    mock_embed_text_oai.assert_called_once()

    mock_embed_text_oai.reset_mock(); caplog.clear()
    long_original_text = " a" * 8200; short_compressed_text = "short"
    results_long = metric.evaluate(original_text=long_original_text, compressed_text=short_compressed_text)
    assert openai_model_name not in results_long["embedding_similarity"]
    mock_embed_text_oai.assert_not_called()
    assert any(record.levelname == "WARNING" and f"Input exceeds model_max_length for {openai_model_name}; skipping" in record.message for record in caplog.records)


def test_hf_tokenizer_max_length_and_skipping(monkeypatch, caplog):
    hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    metric = MultiModelEmbeddingSimilarityMetric(model_names=[hf_model_name])
    original_text = "original text for hf"; compressed_text = "compressed text for hf"
    expected_orig_tokens_normal = 100; expected_comp_tokens_normal = 50;
    expected_comp_tokens_final_normal = 50

    mock_ep_embed = MagicMock(return_value=[[0.5, 0.6], [0.7, 0.8]])
    monkeypatch.setattr(ep, "embed_text", mock_ep_embed)
    mock_tu_token_count = MagicMock()
    monkeypatch.setattr('compact_memory.validation.embedding_metrics.token_utils.token_count', mock_tu_token_count)
    mock_autotokenizer_class = MagicMock(spec=transformers.AutoTokenizer)
    mock_hf_tokenizer_instance = MagicMock()
    mock_autotokenizer_class.from_pretrained.return_value = mock_hf_tokenizer_instance
    monkeypatch.setattr(transformers, "AutoTokenizer", mock_autotokenizer_class)

    # Scenario 1: Within Limits
    mock_hf_tokenizer_instance.model_max_length = 512
    mock_tu_token_count.side_effect = [expected_orig_tokens_normal, expected_comp_tokens_normal, expected_comp_tokens_final_normal]
    results_scen1 = metric.evaluate(original_text=original_text, compressed_text=compressed_text)
    assert hf_model_name in results_scen1["embedding_similarity"]
    model_results_scen1 = results_scen1["embedding_similarity"][hf_model_name]
    assert np.isclose(model_results_scen1["similarity"], np.dot([0.5, 0.6], [0.7, 0.8]))
    assert model_results_scen1["token_count"] == expected_comp_tokens_final_normal
    mock_ep_embed.assert_called_once(); mock_autotokenizer_class.from_pretrained.assert_called_with(hf_model_name)

    # Scenario 2: Exceeds Limits (original_text too long)
    mock_ep_embed.reset_mock(); mock_autotokenizer_class.from_pretrained.reset_mock(); caplog.clear()
    mock_hf_tokenizer_instance.model_max_length = 512
    mock_tu_token_count.side_effect = [600, expected_comp_tokens_normal]
    results_scen2 = metric.evaluate(original_text=original_text, compressed_text=compressed_text)
    assert hf_model_name not in results_scen2["embedding_similarity"]
    mock_ep_embed.assert_not_called()
    assert any(record.levelname == "WARNING" and f"Input exceeds model_max_length for {hf_model_name}; skipping" in record.message for record in caplog.records)

    # Scenario 2b: Exceeds Limits (compressed_text too long)
    mock_ep_embed.reset_mock(); mock_autotokenizer_class.from_pretrained.reset_mock(); caplog.clear()
    mock_hf_tokenizer_instance.model_max_length = 512
    mock_tu_token_count.side_effect = [expected_orig_tokens_normal, 600]
    results_scen2b = metric.evaluate(original_text=original_text, compressed_text=compressed_text)
    assert hf_model_name not in results_scen2b["embedding_similarity"]
    mock_ep_embed.assert_not_called()
    assert any(record.levelname == "WARNING" and f"Input exceeds model_max_length for {hf_model_name}; skipping" in record.message for record in caplog.records)

    # Scenario 3: Invalid or Missing model_max_length
    invalid_max_lens = {"is_none": None, "is_zero": 0, "is_negative": -100, "is_str": "not-an-int", "attr_missing": 'MAGIC_ATTR_MISSING_SENTINEL'}
    for case_name, max_len_val in invalid_max_lens.items():
        mock_ep_embed.reset_mock(); mock_autotokenizer_class.from_pretrained.reset_mock(); caplog.clear(); mock_tu_token_count.reset_mock()
        current_mock_hf_tokenizer_instance = MagicMock()
        if max_len_val == 'MAGIC_ATTR_MISSING_SENTINEL':
            if hasattr(current_mock_hf_tokenizer_instance, 'model_max_length'): del current_mock_hf_tokenizer_instance.model_max_length
        else: current_mock_hf_tokenizer_instance.model_max_length = max_len_val
        mock_autotokenizer_class.from_pretrained.return_value = current_mock_hf_tokenizer_instance
        mock_tu_token_count.side_effect = [expected_comp_tokens_final_normal]
        results_scen3 = metric.evaluate(original_text=original_text, compressed_text=compressed_text)
        assert hf_model_name in results_scen3["embedding_similarity"], f"Failed for case: {case_name}"
        model_results_scen3 = results_scen3["embedding_similarity"][hf_model_name]
        assert np.isclose(model_results_scen3["similarity"], np.dot([0.5, 0.6], [0.7, 0.8])), f"Failed for case: {case_name}"
        assert model_results_scen3["token_count"] == expected_comp_tokens_final_normal, f"Failed for case: {case_name}"
        mock_ep_embed.assert_called_once()
        assert not any(record.levelname == "WARNING" and "Input exceeds model_max_length" in record.message for record in caplog.records)


def test_tokenizer_load_failure(monkeypatch, caplog):
    bad_hf_model_name = "completely-invalid/non-existent-hf-model"
    bad_openai_model_name = "openai/non-existent-openai-model"
    bad_openai_base_name = "non-existent-openai-model"
    valid_hf_model_name = "valid-hf-model/dummy-hf"

    model_names = [bad_hf_model_name, bad_openai_model_name, valid_hf_model_name]
    metric = MultiModelEmbeddingSimilarityMetric(model_names=model_names)
    original_text = "text a"; compressed_text = "text b"

    mock_general_embed_text = MagicMock(return_value=[[0.1,0.1],[0.2,0.2]])
    monkeypatch.setattr(ep, "embed_text", mock_general_embed_text)

    mock_general_tu_token_count = MagicMock(side_effect=[10, 5, 5])
    monkeypatch.setattr('compact_memory.validation.embedding_metrics.token_utils.token_count', mock_general_tu_token_count)

    mock_valid_hf_tokenizer_instance = MagicMock()
    mock_valid_hf_tokenizer_instance.model_max_length = 128

    def hf_autotokenizer_side_effect(model_name, **kwargs):
        if model_name == bad_hf_model_name:
            raise OSError("HF tokenizer load error for bad_hf_model")
        elif model_name == valid_hf_model_name:
            return mock_valid_hf_tokenizer_instance
        pytest.fail(f"Unexpected call to AutoTokenizer.from_pretrained with {model_name}")

    mock_hf_autotokenizer_class = MagicMock(spec=transformers.AutoTokenizer)
    mock_hf_autotokenizer_class.from_pretrained.side_effect = hf_autotokenizer_side_effect
    monkeypatch.setattr(transformers, "AutoTokenizer", mock_hf_autotokenizer_class)

    def mock_tiktoken_efm_for_failure(model_name_arg):
        if model_name_arg == bad_openai_base_name:
            raise ValueError("Primary tiktoken error for bad_openai_base_name")
        return MagicMock(spec=tiktoken.Encoding)

    def mock_tiktoken_ge_for_failure(encoding_name_arg):
        if encoding_name_arg == "gpt2":
            raise ValueError("Fallback gpt2 error for bad_openai_base_name")
        return MagicMock(spec=tiktoken.Encoding)

    with patch('tiktoken.encoding_for_model', side_effect=mock_tiktoken_efm_for_failure), \
         patch('tiktoken.get_encoding', side_effect=mock_tiktoken_ge_for_failure):

        results = metric.evaluate(original_text=original_text, compressed_text=compressed_text)

    assert bad_hf_model_name not in results["embedding_similarity"]
    assert bad_openai_model_name not in results["embedding_similarity"]
    assert valid_hf_model_name in results["embedding_similarity"]

    valid_model_results = results["embedding_similarity"][valid_hf_model_name]
    assert valid_model_results["token_count"] == 5
    assert np.isclose(valid_model_results["similarity"], np.dot([0.1,0.1],[0.2,0.2]))

    log_text = caplog.text
    # Corrected log message check (without "OSError: ")
    assert f"Failed loading tokenizer for {bad_hf_model_name}: HF tokenizer load error for bad_hf_model" in log_text
    assert f"Failed loading tokenizer for {bad_openai_model_name}: Fallback gpt2 error for bad_openai_base_name" in log_text

    mock_hf_autotokenizer_class.from_pretrained.assert_any_call(valid_hf_model_name)
    was_embed_text_called_for_valid_model = False
    for call in mock_general_embed_text.call_args_list:
        _, kwargs = call
        if kwargs.get("model_name") == valid_hf_model_name:
            was_embed_text_called_for_valid_model = True
            break
    assert was_embed_text_called_for_valid_model
    assert mock_general_tu_token_count.call_count == 3
