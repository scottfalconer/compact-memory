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

`MultiEmbeddingSimilarityMetric` (metric ID `embedding_similarity_multi`)
accepts multiple SentenceTransformer model IDs via a `model_names` list. The
metric reports an averaged `semantic_similarity` score along with individual
scores for each model and a `token_count` for the evaluated pair. If the token
count exceeds the configured `max_tokens` limit the pair is skipped.

```python
from compact_memory.validation.embedding_metrics import MultiEmbeddingSimilarityMetric

metric = MultiEmbeddingSimilarityMetric(
    model_names=["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"],
    max_tokens=8192,
)
scores = metric.evaluate(original_text="hello", compressed_text="hi")
# {
#   "semantic_similarity": 0.98,
#   "all-MiniLM-L6-v2": 0.99,
#   "multi-qa-mpnet-base-dot-v1": 0.97,
#   "token_count": 4
# }
```

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
