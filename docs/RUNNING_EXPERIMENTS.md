# Running Experiments

This document explains how to use the Compact Memory experimentation framework to evaluate and compare different `BaseCompressionEngine` implementations. It covers configuring experiments, specifying validation metrics, and understanding the results.

Experiments are configured with `ExperimentConfig` or `ResponseExperimentConfig`.
Validation metrics are listed in the `validation_metrics` field:

```python
cfg = ResponseExperimentConfig(
    dataset=Path("dialogues.yaml"),
    param_grid=[{"config_prompt_num_forced_recent_turns": 1}],
    validation_metrics=[
        {"id": "rouge_hf", "params": {"rouge_types": ["rouge1"]}},
        {"id": "exact_match", "params": {}},
    ],
)
```

During the run each metric is instantiated and its scores averaged across the
Beyond task-specific accuracy, experiments should ideally capture efficiency metrics. The `CompressionTrace` object (see `compact_memory/engines/__init__.py`) is designed to hold such details. Key metrics include:
    * Final compressed prompt token count.
    * Original uncompressed token count.
    * Processing time of the compression engine.
    * Compression ratio achieved.
dataset. Results are returned as a list of dictionaries containing the parameter
set and metric scores.
