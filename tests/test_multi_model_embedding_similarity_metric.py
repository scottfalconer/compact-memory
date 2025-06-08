import numpy as np
from unittest.mock import MagicMock
from compact_memory.validation.embedding_metrics import (
    MultiModelEmbeddingSimilarityMetric,
)


def test_multi_model_embedding_similarity_basic(patch_embedding_model):
    metric = MultiModelEmbeddingSimilarityMetric(model_names=["dummy-model"])
    scores = metric.evaluate(original_text="hello", compressed_text="hello")
    data = scores["embedding_similarity"]["dummy-model"]
    assert np.isclose(data["similarity"], 1.0)
    assert data["token_count"] == 1


def test_multi_model_embedding_similarity_skip_long(patch_embedding_model):
    metric = MultiModelEmbeddingSimilarityMetric(model_names=["dummy-model"])
    long_text = " ".join("t" + str(i) for i in range(200))
    scores = metric.evaluate(original_text=long_text, compressed_text=long_text)
    assert scores["embedding_similarity"] == {}


def test_multi_model_embedding_similarity_multiple_hf_models(monkeypatch):
    # 1. Patch target functions
    mock_embed_text = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.ep.embed_text", mock_embed_text
    )
    mock_token_count = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.token_utils.token_count",
        mock_token_count,
    )

    # 2. Define mock behaviors
    def embed_text_side_effect(
        texts, model_name, encoder_model=None, tokenizer=None, **kwargs
    ):
        print(
            f"embed_text_side_effect: texts='{texts}', model_name='{model_name}', encoder_model is None: {encoder_model is None}, tokenizer is None: {tokenizer is None}"
        )

        if (
            len(texts) == 2
        ):  # Batch call from the metric directly: ep.embed_text([text_a, text_b], model_name=name)
            original_text_val = texts[0]
            compressed_text_val = texts[1]

            embedding_original = [0.0, 0.0]  # Default
            embedding_compressed = [0.0, 0.0]  # Default

            if model_name == "hf_model_1":
                if original_text_val == "original text":
                    embedding_original = [0.1, 0.2]
                if compressed_text_val == "compressed text":
                    embedding_compressed = [0.3, 0.4]
            elif model_name == "hf_model_2":
                if original_text_val == "original text":
                    embedding_original = [0.5, 0.6]
                if compressed_text_val == "compressed text":
                    embedding_compressed = [0.7, 0.8]

            print(
                f"Batch call for {model_name}: returning {[embedding_original, embedding_compressed]}"
            )
            return [embedding_original, embedding_compressed]

        elif len(texts) == 1:  # Individual call (likely from EmbedderPipeline instance)
            text = texts[0]
            # These calls might be for text splitting logic or other checks,
            # not necessarily the ones directly used for the final similarity score.
            # Provide consistent embeddings.
            if model_name == "hf_model_1":
                if text == "original text":
                    print(
                        f"Individual call for {model_name}, '{text}': returning [[0.1, 0.2]]"
                    )
                    return [[0.1, 0.2]]
                elif text == "compressed text":
                    print(
                        f"Individual call for {model_name}, '{text}': returning [[0.3, 0.4]]"
                    )
                    return [[0.3, 0.4]]
            elif model_name == "hf_model_2":
                if text == "original text":
                    print(
                        f"Individual call for {model_name}, '{text}': returning [[0.5, 0.6]]"
                    )
                    return [[0.5, 0.6]]
                elif text == "compressed text":
                    print(
                        f"Individual call for {model_name}, '{text}': returning [[0.7, 0.8]]"
                    )
                    return [[0.7, 0.8]]

            print(
                f"Individual call for {model_name}, '{text}': defaulting to [[0.0, 0.0]]"
            )
            return [[0.0, 0.0]]

        # Fallback for unexpected number of texts
        print(f"Unexpected number of texts ({len(texts)}) for {model_name}: defaulting")
        return [[0.0, 0.0]] * len(texts) if texts else []

    mock_embed_text.side_effect = embed_text_side_effect

    # State for token_count_side_effect
    # Maps tokenizer object ID to a pre-assigned model name ("hf_model_1" or "hf_model_2")
    tokenizer_object_to_model_name_map = {}
    # Assumes model_names in metric are processed in this order for initial mapping
    model_names_in_processing_order = ["hf_model_1", "hf_model_2"]

    # The call from source code is token_utils.token_count(tokenizer_object, text_string)
    # tokenizer_object is a DummyTokenizer instance. Log shows these are different for different models.
    def token_count_side_effect(tokenizer_dummy_obj, text_content_str, **kwargs):
        nonlocal tokenizer_object_to_model_name_map, model_names_in_processing_order

        obj_id = id(tokenizer_dummy_obj)
        current_model_name = tokenizer_object_to_model_name_map.get(obj_id)

        if current_model_name is None:  # First time seeing this tokenizer object
            # Assign it the next available model name based on processing order
            if len(tokenizer_object_to_model_name_map) < len(
                model_names_in_processing_order
            ):
                current_model_name = model_names_in_processing_order[
                    len(tokenizer_object_to_model_name_map)
                ]
                tokenizer_object_to_model_name_map[obj_id] = current_model_name
            else:
                # Should not happen if only 2 models are used and processed once for tokenizers
                current_model_name = "unknown_model_obj_id_" + str(obj_id)

        print(
            f"token_count_side_effect: text='{text_content_str}', model='{current_model_name}' (from obj id {obj_id})"
        )

        count_to_return = 0
        # We only care about returning specific counts for "compressed text" for the models under test.
        # Other calls (e.g., for "original text", or from text_is_too_long path if args are swapped) should get a default.
        if text_content_str == "compressed text":
            if current_model_name == "hf_model_1":
                print(f"Returning 10 for {current_model_name} 'compressed text'")
                count_to_return = 10
            elif current_model_name == "hf_model_2":
                print(f"Returning 20 for {current_model_name} 'compressed text'")
                count_to_return = 20
            else:
                print(
                    f"Unknown model '{current_model_name}' for 'compressed text', returning 0"
                )
                count_to_return = 0
        else:
            # For "original text", or if text_content_str is actually a tokenizer obj due to swapped args from text_is_too_long
            print(
                f"Text ('{text_content_str}') not 'compressed text' for model '{current_model_name}', or args swapped; returning 0"
            )
            count_to_return = 0

        return count_to_return

    mock_token_count.side_effect = token_count_side_effect

    # 3. Instantiate MultiModelEmbeddingSimilarityMetric
    metric = MultiModelEmbeddingSimilarityMetric(
        model_names=["hf_model_1", "hf_model_2"]
    )

    # 4. Call evaluate
    result = metric.evaluate(
        original_text="original text", compressed_text="compressed text"
    )

    # 5. Assertions
    assert "embedding_similarity" in result
    es_results = result["embedding_similarity"]

    assert "hf_model_1" in es_results
    assert np.isclose(
        es_results["hf_model_1"]["similarity"], np.dot([0.1, 0.2], [0.3, 0.4])
    )  # 0.03 + 0.08 = 0.11
    assert es_results["hf_model_1"]["token_count"] == 10

    assert "hf_model_2" in es_results
    assert np.isclose(
        es_results["hf_model_2"]["similarity"], np.dot([0.5, 0.6], [0.7, 0.8])
    )  # 0.35 + 0.48 = 0.83
    assert es_results["hf_model_2"]["token_count"] == 20


def test_multi_model_openai_embedding_failure(monkeypatch):
    # 1. Patch necessary functions/methods
    mock_ep_embed_text = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.ep.embed_text",
        mock_ep_embed_text,
    )
    mock_token_utils_token_count = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.token_utils.token_count",
        mock_token_utils_token_count,
    )
    mock_get_tokenizer = MagicMock()
    monkeypatch.setattr(
        "compact_memory.validation.embedding_metrics.MultiModelEmbeddingSimilarityMetric._get_tokenizer",
        mock_get_tokenizer,
    )

    # 2. Define mock behaviors
    openai_model_name = "openai/text-embedding-ada-002"

    # Mock for _get_tokenizer
    mock_openai_tokenizer = MagicMock()
    mock_openai_tokenizer.model_max_length = 8192
    mock_get_tokenizer.return_value = mock_openai_tokenizer

    # Mock for ep.embed_text (global embedder)
    def embed_text_side_effect(
        texts, model_name, encoder_model=None, tokenizer=None, **kwargs
    ):
        print(f"ep.embed_text called with: texts='{texts}', model_name='{model_name}'")
        if model_name == openai_model_name:
            # Simulate failure for OpenAI model by returning zero embeddings for the batch call
            print(
                f"Simulating embedding failure for {openai_model_name}, returning zero embeddings."
            )
            # It's a batch call from the metric: expects list of 2 embeddings
            return [[0.0, 0.0], [0.0, 0.0]]
        # Fallback for other models/calls if any (not expected in this test)
        return [[0.0, 0.0]] * len(texts) if texts else []

    mock_ep_embed_text.side_effect = embed_text_side_effect

    # Mock for token_utils.token_count
    # This mock needs to handle two call signatures due to how it's used in the codebase:
    # 1. Direct call: token_utils.token_count(tokenizer_object, text_string)
    # 2. Via Tokenizer.token_count: token_utils.token_count(text_string, hf_dummy_tokenizer, model_name_string)
    def token_count_side_effect(*args, **kwargs):
        # Check the type of the first argument to differentiate call patterns
        arg1 = args[0]

        if isinstance(
            arg1, str
        ):  # Call from Tokenizer.token_count(text, hf_dummy, model_name)
            text_content_str = arg1
            # arg2 is hf_dummy_tokenizer, arg3 is model_name_str (passed as kwarg or positional)
            # This path is mainly for text_is_too_long, not for the primary token count assertion.
            # We can return a default value or make it more specific if needed.
            print(
                f"token_utils.token_count (via Tokenizer.token_count): text='{text_content_str}', args='{args}'"
            )
            return 0  # Default for this path
        else:  # Direct call: token_utils.token_count(tokenizer_object, text_string)
            tokenizer_obj = arg1
            text_content_str = args[1]
            print(
                f"token_utils.token_count (direct): text='{text_content_str}', tokenizer_obj='{tokenizer_obj}'"
            )
            if (
                tokenizer_obj == mock_openai_tokenizer
                and text_content_str == "compressed text"
            ):
                print(f"Returning 30 for OpenAI tokenizer and 'compressed text'")
                return 30
            return 0  # Default for other direct calls

    mock_token_utils_token_count.side_effect = token_count_side_effect

    # 3. Instantiate MultiModelEmbeddingSimilarityMetric
    metric = MultiModelEmbeddingSimilarityMetric(model_names=[openai_model_name])

    # 4. Call evaluate
    result = metric.evaluate(
        original_text="original text", compressed_text="compressed text"
    )

    # 5. Assertions
    assert "embedding_similarity" in result
    es_results = result["embedding_similarity"]

    assert openai_model_name in es_results
    openai_result = es_results[openai_model_name]

    assert openai_result["token_count"] == 30
    # If ep.embed_text returned [[0.0,0.0], [0.0,0.0]], then similarity = dot([0,0],[0,0]) = 0.0
    assert np.isclose(openai_result["similarity"], 0.0)
