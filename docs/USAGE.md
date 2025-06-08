# Compact Memory Usage Guide

This guide shows how to use the Compact Memory toolkit from the command line and Python.

## Basic CLI Usage

The CLI automatically registers the experimental compression engines, so options like `first_last` work without extra setup.

Compress a text file using the `first_last` compression engine with a token budget of 100:

```bash
compact-memory compress --file "path/to/your_document.txt" --engine first_last --budget 100
```

Compress a string directly:

```bash
compact-memory compress --text "This is a very long string that needs to be much shorter to fit into my LLM's context window." --engine truncate --budget 20
```

You can also pipe input via standard input:

```bash
cat notes.txt | compact-memory compress --text - --engine truncate --budget 20
```

Write the output to a file with `-o`:

```bash
compact-memory compress --file "path/to/your_document.txt" --engine first_last --budget 100 -o "path/to/compressed_output.txt"
```

### Compressing a Directory

When you need to compress all text files within a directory (and optionally its subdirectories), `compact-memory` offers a directory compression mode.

**Behavior:**

*   All files matching the specified pattern (default: `*.txt`) within the input directory are read.
*   If the `--recursive` (or `-r`) option is used, files in subdirectories are also included.
*   The content of these files is **concatenated into a single body of text**.
*   This combined text is then compressed using the chosen engine and budget.
*   The result is a **single output file** named `compressed_output.txt`.

**Output Path:**

*   If you specify an `--output-dir`, the `compressed_output.txt` file will be saved in that directory.
*   If no `--output-dir` is provided, `compressed_output.txt` will be saved directly within the input directory (`--dir`).
*   The `--output` (or `-o`) option **cannot be used** with `--dir`.

**Examples:**

1.  Compress all `.txt` files in `path/to/your_text_files/` and save the result to `path/to/output_directory/compressed_output.txt`:

    ```bash
    compact-memory compress --dir "path/to/your_text_files/" --engine first_last --budget 200 --output-dir "path/to/output_directory/"
    ```
    *(Expected output: `path/to/output_directory/compressed_output.txt`)*

2.  Compress all `.md` files recursively in `project_docs/` and save `compressed_output.txt` into `project_docs/`:

    ```bash
    compact-memory compress --dir "project_docs/" --pattern "*.md" --recursive --engine summarization_engine --budget 500
    ```
    *(Expected output: `project_docs/compressed_output.txt`)*


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

You can also call compression engines programmatically:

```python
from compact_memory import get_compression_engine
from compact_memory.token_utils import get_tokenizer

engine = get_compression_engine("first_last")()
text_to_compress = "This is a very long document that we want to summarize."
compressed, _ = engine.compress(text_to_compress, llm_token_budget=50)
print(compressed.text)
```

The [CLI reference](cli_reference.md) and [API reference](api_reference.md) contain additional details.
