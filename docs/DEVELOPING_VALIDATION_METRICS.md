# Developing Validation Metrics

This document guides developers and researchers on creating custom `ValidationMetric` classes to evaluate the quality and utility of compressed memory produced by different `BaseCompressionEngine` implementations.

Validation metrics assess how well an LLM performs when using compressed memory.
All metrics must subclass `ValidationMetric` and register themselves with the
registry for easy lookup.

## ValidationMetric ABC

```python
from compact_memory.validation.metrics_abc import ValidationMetric

class MyMetric(ValidationMetric):
    metric_id = "my_metric"

    def evaluate(self, llm_response: str, reference_answer: str, **kwargs):
        ...  # return {"score_name": value}
```

Use `register_validation_metric` to make the metric discoverable:

```python
from compact_memory.registry import register_validation_metric

register_validation_metric(MyMetric.metric_id, MyMetric)
```

## Hugging Face evaluate Metrics

Metrics backed by the [`evaluate`](https://github.com/huggingface/evaluate)
library can extend `HFValidationMetric` which handles loading the metric.
Install the optional dependency with `pip install compact-memory[metrics]`:

```python
from compact_memory.validation.hf_metrics import HFValidationMetric

class RougeMetric(HFValidationMetric):
    metric_id = "rouge_hf"

    def __init__(self, **params):
        super().__init__("rouge", **params)
```

Metrics are selected by `metric_id` and optional initialisation parameters.
Use `list_validation_metrics()` from `compact_memory.validation.registry` to
programmatically retrieve the available metric IDs.

## Embedding-based Metrics

`EmbeddingSimilarityMetric` computes cosine similarity between sentence
embeddings. It relies on the optional `sentence-transformers` dependency.
Install it with `pip install compact-memory[embedding]`.

```python
from compact_memory.validation.embedding_metrics import EmbeddingSimilarityMetric

metric = EmbeddingSimilarityMetric(model_name="all-MiniLM-L6-v2")
scores = metric.evaluate(original_text="a", compressed_text="b")
```

### Multi‑model Similarity

`MultiModelEmbeddingSimilarityMetric` (metric ID `multi_model_embedding_similarity`)
evaluates text similarity using multiple embedding models. It accepts a `model_names`
list, which can include HuggingFace SentenceTransformer identifiers (e.g.,
`"sentence-transformers/all-MiniLM-L6-v2"`) or OpenAI model identifiers (e.g.,
`"openai/text-embedding-ada-002"`).

For each model, it calculates the cosine similarity between the two input texts
and also determines the token count of the second text (e.g., `compressed_text`
or `llm_response`) using that specific model's tokenizer.

If `model_names` is not provided during instantiation, a default list of diverse
SentenceTransformer models is used. The metric will skip evaluation for any model
if either input text exceeds that model's maximum token limit, logging a warning
in such cases.

```python
from compact_memory.validation.embedding_metrics import MultiModelEmbeddingSimilarityMetric

metric = MultiModelEmbeddingSimilarityMetric(
    model_names=["sentence-transformers/all-MiniLM-L6-v2", "openai/text-embedding-ada-002"]
)
scores = metric.evaluate(original_text="hello world", compressed_text="hello there")
# Example Output:
# {
#   "embedding_similarity": {
#     "sentence-transformers/all-MiniLM-L6-v2": {
#       "similarity": 0.85, # Example value
#       "token_count": 2    # Example value for "hello there"
#     },
#     "openai/text-embedding-ada-002": {
#       "similarity": 0.88, # Example value
#       "token_count": 2    # Example value for "hello there"
#     }
#   }
# }
```

**Prerequisites:**
- For HuggingFace SentenceTransformer models: `sentence-transformers` library.
- For OpenAI models: `tiktoken` library and the `OPENAI_API_KEY` environment variable must be set.
You can install necessary extras with `pip install compact-memory[embedding]`.

**Performance Note:** Using more, or larger, embedding models will increase evaluation time. OpenAI models also involve API calls which can add latency and costs.

## LLM Judge Metric

`LLMJudgeMetric` queries an OpenAI chat model to score text pairs. The metric
caches results in memory to avoid repeated API calls.

```python
from compact_memory.validation.llm_judge_metric import LLMJudgeMetric

metric = LLMJudgeMetric(model_name="gpt-4")
score = metric.evaluate(llm_response=model_answer, reference_answer=truth)
```

Ensure your `OPENAI_API_KEY` environment variable is set before using this
metric. The metric raises a `RuntimeError` if no API key is found.
