# Developing Validation Metrics

This document guides developers and researchers on creating custom `ValidationMetric` classes to evaluate the quality and utility of compressed memory produced by different `CompressionStrategy` implementations. These metrics are crucial for the Compact Memory experimentation framework.

Validation metrics assess how well an LLM performs when using compressed memory.
All metrics must subclass `ValidationMetric` and register themselves with the
registry so experiment configurations can reference them.

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

When running experiments, metrics are selected by `metric_id` and optional
initialisation parameters.
