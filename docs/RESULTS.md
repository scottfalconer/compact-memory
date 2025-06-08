# Experiment Results

This document summarizes small-scale benchmarks run with Compact Memory. We evaluated three compression engines on the repository's test dialogue datasets using a dummy local language model (as in the test suite).

## Response Experiment
Dataset: `tests/data/response_dialogues.yaml`
Metric: `exact_match`

| Engine      | Avg Prompt Tokens | Exact Match |
|---------------|------------------|-------------|
| none          | 37.0             | 1.0         |
| importance    | 37.0             | 1.0         |
| first_last    | 37.0             | 1.0         |

## History Experiment
Dataset: `tests/data/history_dialogues.yaml`

| Params | Hit Rate |
|--------|----------|
| `config_prompt_num_forced_recent_turns=1` | 1.0 |

## Compression Example
Using `tests/data/constitution.txt` with a token budget of 100, the `FirstLastEngine` achieved a compression ratio of approximately 0.05.

These quick measurements illustrate how to evaluate engine behavior. For larger-scale benchmarks (e.g., long‑context QA or summarization), the same configuration patterns apply.

## Engine Metrics Script (`examples/collect_engine_metrics.py`)

The `examples/collect_engine_metrics.py` script is a utility to gather `compression_ratio` and `embedding_similarity` metrics for all available (registered) compression engines in the `compact_memory` library.

By default, the script uses the text found in `tests/data/constitution.txt` as the input data for these evaluations.

The script outputs a JSON file (named `engine_metrics.json` by default) containing the calculated metrics for each engine. An illustrative example of this output format is:

```json
{
  "none": {
    "compression_ratio": 1.0,
    "embedding_similarity": 1.0
  },
  "first_last": {
    "compression_ratio": 0.05,  // Example value
    "embedding_similarity": 0.85 // Example value
  }
  // ... other engines
}
```

### Important Considerations

-   The `embedding_similarity` metric produced by this script is intended to use a real sentence transformer model to provide meaningful semantic similarity scores.
-   The script has been updated to facilitate this by removing the `MockEncoder` override.
-   However, running the script with a real model requires an environment with sufficient disk space and network access to download and install the model and its dependencies (e.g., `sentence-transformers`, `torch`, `nvidia-cudnn-cu12` if using CUDA).
-   If such resources are unavailable (as was the case during a recent review attempt which failed due to disk space limitations), the `embedding_similarity` scores in any generated `engine_metrics.json` might reflect a fallback or previously generated placeholder values (e.g., from `MockEncoder`) and should not be considered indicative of true semantic similarity until the script can be successfully run with a real model.
