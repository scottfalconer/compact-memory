# Evaluation Guide

Compact Memory no longer ships with a built-in experimentation framework. Instead, use external tools to benchmark compression strategies.

## Recommended Tools

- **Promptfoo** – CLI for A/B testing prompts, supports LLM-as-a-judge workflows.
- **UpTrain** – Code-first framework providing factuality and recall metrics out of the box.
- **Helicone** – UI-based prompt monitoring and scoring solution.
- **OpenAI Evals** – For thorough ground-truth QA benchmarks.

## How to Use With Compact Memory

1. Generate compressed context using the CLI or Python API:
   ```bash
   compact-memory compress input.txt output.txt --strategy prototype
   ```
2. Feed the compressed output into your evaluation tool as part of a prompt template.
3. Compare results against the uncompressed baseline.
4. Store test cases in JSONL or CSV format for easy iteration.

## Quick Start

See [`examples/eval_promptfoo.yaml`](../examples/eval_promptfoo.yaml) for a minimal Promptfoo configuration using the sample datasets under `examples/eval/`.
