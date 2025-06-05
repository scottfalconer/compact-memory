# Experiment Results

This document summarizes small-scale benchmarks run with the Compact Memory experimentation framework. We evaluated three compression strategies on the repository's test dialogue datasets using a dummy local language model (as in the test suite).

## Response Experiment
Dataset: `tests/data/response_dialogues.yaml`
Metric: `exact_match`

| Strategy      | Avg Prompt Tokens | Exact Match |
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
Using `tests/data/constitution.txt` with a token budget of 100, the `FirstLastStrategy` achieved a compression ratio of approximately 0.05.

These quick experiments illustrate how to measure strategy behavior with the provided framework. For larger-scale benchmarks (e.g., long‑context QA or summarization), the same configuration patterns apply.
