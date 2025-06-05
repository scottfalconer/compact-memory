# Compact Memory Usage Guide

This guide shows how to use the Compact Memory toolkit from the command line and Python.

## Basic CLI Usage

The CLI automatically registers the experimental compression strategies, so options like `first_last` work without extra setup.

Compress a text file using the `first_last` compression strategy with a token budget of 100:

```bash
compact-memory compress --file "path/to/your_document.txt" --strategy first_last --budget 100
```

Compress a string directly:

```bash
compact-memory compress --text "This is a very long string that needs to be much shorter to fit into my LLM's context window." --strategy truncate --budget 20
```

You can also pipe input via standard input:

```bash
cat notes.txt | compact-memory compress --text - --strategy truncate --budget 20
```

Write the output to a file with `-o`:

```bash
compact-memory compress --file "path/to/your_document.txt" -s first_last -b 100 -o "path/to/compressed_output.txt"
```

### Using Compressed Output in an LLM Prompt

The compressed text can be inserted into prompts directly. For example, if `compressed_output.txt` contains the summary you produced above you might write:

```
System: You are a helpful assistant.
User: Based on the following summary, what are the main benefits of Compact Memory?

Summary:
---
$(cat compressed_output.txt)
---

Assistant:
```

## Python API Example

You can also call compression strategies programmatically:

```python
from compact_memory.compression import get_compression_strategy
from compact_memory.token_utils import get_tokenizer

strategy = get_compression_strategy("first_last")()
text_to_compress = "This is a very long document that we want to summarize."
compressed, _ = strategy.compress(text_to_compress, llm_token_budget=50)
print(compressed.text)
```

The [CLI reference](docs/cli_reference.md) and [API reference](docs/api_reference.md) contain additional details.
