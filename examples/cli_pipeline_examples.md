# Command-Line Examples for PipelineEngine

This document provides examples of how to use the `PipelineEngine` with the `compact-memory compress` command-line interface. The `PipelineEngine` allows you to chain multiple compression engines together, where the output of one engine becomes the input for the next.

## Example 1: Simple Pipeline (No Compression -> First/Last Selection)

**Description:**
This example demonstrates a basic pipeline. The first stage uses `NoCompressionEngine`, which, if the budget is sufficient for the input text, passes the text through unaltered. The second stage uses `FirstLastEngine` to select a specific number of units (e.g., words or sentences, depending on its internal tokenizer and logic) from the beginning and end of the text.

**Command:**
```bash
compact-memory compress --engine pipeline \
  --pipeline-config '{
    "engines": [
      {"engine_name": "NoCompressionEngine", "engine_params": {}},
      {"engine_name": "FirstLastEngine", "engine_params": {"first_n": 10, "last_n": 5}}
    ]
  }' \
  --text "This is a very long example sentence that we want to process using a pipeline. It has many words, and we will see how the FirstLastEngine truncates it after the NoCompressionEngine does nothing effectively. The middle part should be gone." \
  --budget 20
```

**Expected Outcome:**
The `NoCompressionEngine` will pass the full text to the `FirstLastEngine` (assuming the initial text within the overall `--budget` allows, though `NoCompressionEngine` itself doesn't strictly enforce a budget in the same way other engines might; it's more of a pass-through or minimal change engine). The `FirstLastEngine` will then process this text. If its internal logic splits by words, the output would roughly be the first 10 words and the last 5 words of the input sentence, concatenated. The actual output depends on the `FirstLastEngine`'s tokenization and unit handling. The `--budget 20` applies to the final output of the pipeline.

## Example 2: Pipeline with Engine-Specific Parameters (Stopword Pruning -> First/Last Selection)

**Description:**
This pipeline first processes the text with `StopwordPrunerEngine` to remove common words (stopwords) based on the specified language. The output of this pruning step (text with stopwords removed) is then passed to `FirstLastEngine`. `FirstLastEngine` then selects the first and last few units from the pruned text. The `llm_token_budget` within `FirstLastEngine`'s `engine_params` is an example of a parameter that might be specific to how that engine further constrains its own output, separate from the global `--budget` which the pipeline aims for as a whole.

**Command:**
```bash
compact-memory compress --engine pipeline \
  --pipeline-config '{
    "engines": [
      {"engine_name": "StopwordPrunerEngine", "engine_params": {"lang": "english"}},
      {"engine_name": "FirstLastEngine", "engine_params": {"first_n": 3, "last_n": 3, "llm_token_budget": 30}}
    ]
  }' \
  --text "This is an example sentence with many common words like the, is, an, and with. We will try to prune stopwords and then take the first and last few important words that remain." \
  --budget 30
```

**Expected Outcome:**
First, the `StopwordPrunerEngine` removes common English stopwords (e.g., "this", "is", "an", "with", "the"). The resulting text, now shorter and denser with keywords, is then processed by `FirstLastEngine`. This engine will take the first 3 and last 3 units (words/sentences) from the stopword-pruned text. The final output will be a concise version of the sentence, focusing on the less common words from the beginning and end of the pruned version, and aiming to be within the global `--budget` of 30 tokens.

## Note on Engine Availability

The availability of specific compression engines mentioned in these examples (e.g., `NoCompressionEngine`, `FirstLastEngine`, `StopwordPrunerEngine`) depends on your Compact Memory installation and any additionally installed engine plugins. These are often part of the core or example engines provided with the framework.

## Further Information

For more detailed information on using the `compact-memory compress` command and configuring the `PipelineEngine`, please refer to the main documentation, particularly:
*   [CLI Reference](../docs/cli_reference.md#using-the-pipelineengine---engine-pipeline)
*   [Compression Engines Overview](../docs/COMPRESSION_ENGINES.md#g-pipeline-engine-pipeline)
