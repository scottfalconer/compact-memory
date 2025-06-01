# Running Experiments

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
dataset. Results are returned as a list of dictionaries containing the parameter
set and metric scores.
