# Designing Compression Strategies

This guide collects techniques for slicing documents into belief-sized ideas and updating prototypes. A compression strategy can mix and match these steps depending on the data source.
While LLMs with large context windows and RAG are powerful, Compact Memory explores strategies for more deeply processed, long-term, and adaptive memory. The techniques described here aim to create dynamic memory structures that evolve with new information, offering capabilities beyond simple retrieval of verbatim text chunks.

## Segmenting source documents

### 1. Fast first-pass split
- Paragraph or sentence boundaries using regex or spaCy with a small token overlap.
- Fixed token windows that recurse on punctuation (as used by LangChain and Llama-Index).

These methods are fast and require no machine learning, but they can cut ideas in half and generate near duplicates.

### 2. Semantic boundary detection
- TextTiling-style lexical cohesion where a cosine similarity drop marks a topic break.
- Learned models such as CrossFormer or a multi-granular splitter that keeps parent/child links for multi-resolution search.

Boundaries fall on real topic shifts, so retrieval returns fewer irrelevant neighbours.

### 3. LLM assisted proposition extraction
- Prompt the model to list distinct factual statements or extract subject/predicate/object triples.
- Summaries with bullets retain a human-readable gist.
This step can itself be powered by a smaller fine-tuned model, illustrating how learned components plug into a `CompressionStrategy`.

Batching through a local model keeps the cost manageable and only long segments need a full LLM pass.

## User-Defined Chunking

Compact Memory strategies can leverage custom chunking functions to split input text before compression. This provides fine-grained control over how text is segmented, enabling strategies to work with more meaningful units of information.

### The `ChunkFn` Interface

A chunking function, or `ChunkFn`, is a simple callable that takes a single string as input and returns a list of strings (the chunks). The type hint for this interface is:

```python
from typing import Callable, List
ChunkFn = Callable[[str], List[str]]
```

### Example: Custom Chunking Function

You can easily create your own chunking functions. For instance, to split text by double newline characters:

```python
# my_custom_splitters.py
from typing import List

def split_by_double_newline(text: str) -> List[str]:
    """Splits text by occurrences of two newline characters."""
    return text.split('\n\n')
```

### Using Custom Chunking in Python API

To use your custom chunking function with a compression strategy in Python:

```python
from compact_memory.compression import get_compression_strategy
# Assuming your custom function is in my_custom_splitters.py and accessible in PYTHONPATH
# from my_custom_splitters import split_by_double_newline

# Or define it in the same scope:
# def split_by_double_newline(text: str) -> List[str]:
#     return text.split('\n\n')

strategy_id = "first_last" # Example strategy
strategy = get_compression_strategy(strategy_id)()

my_text = "This is the first paragraph.\n\nThis is the second paragraph, separated by a double newline.\n\nAnd a third one."

# Pass the custom chunk_fn to the compress method
# Make sure split_by_double_newline is defined or imported
# compressed_mem, trace = strategy.compress(text=my_text, llm_token_budget=100, chunk_fn=split_by_double_newline)
# print(f"Compressed with custom chunking: {compressed_mem.text}")
```
*(Note: You would need to ensure `split_by_double_newline` is actually defined or imported in the scope where you call `strategy.compress` for the example above to run.)*

### Using Custom Chunking via CLI

You can also specify a custom chunking function when using the `compact-memory compress` command via the `--chunk-script` option. This option takes a string formatted as `'path/to/your/script.py:your_function_name'`.

```bash
# Assuming my_custom_splitters.py is in the current directory or accessible via path
compact-memory compress "input_document.txt" \
    --strategy first_last \
    --budget 100 \
    --chunk-script "my_custom_splitters.py:split_by_double_newline"
```

### Provided Examples

The `examples/chunking.py` file in the Compact Memory repository contains several pre-defined chunking functions that you can use or adapt:
*   `newline_splitter`: Splits text by single newline characters.
*   `tiktoken_fixed_size_splitter`: Splits text into fixed-size chunks using `tiktoken` for tokenization, useful for respecting model token limits.
*   `langchain_recursive_splitter`: A wrapper around LangChain's `RecursiveCharacterTextSplitter`, offering sophisticated recursive splitting.
*   `agentic_split` & `simple_sentences`: More specialized heuristic-based sentence splitters originally used by some internal components.

To use these, you would point `--chunk-script` to `examples/chunking.py` and the respective function name, e.g., `examples/chunking.py:newline_splitter`.

### No Chunking

If no `chunk_fn` (Python API) or `--chunk-script` (CLI) is provided to a strategy, the strategy will typically process the input text as a single, monolithic block. Most strategies will internally wrap the input `text` in a list (e.g., `chunks = [text]`) and proceed with their logic.

### Advanced Chunking Needs

For more advanced text segmentation requirements, consider integrating functions from specialized libraries:
*   **LangChain:** Offers a variety of `TextSplitter` classes (e.g., `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`, `TokenTextSplitter`).
*   **NLTK:** Provides tools like `sent_tokenize` for sentence splitting and `word_tokenize` for word tokenization.
*   **spaCy:** Excellent for robust sentence segmentation and more linguistic processing.
*   **Hugging Face Tokenizers:** Their methods can be used for custom splitting based on subword tokens or other model-specific tokenization rules.

You can wrap these external library functions into a `ChunkFn` compatible interface to use them with Compact Memory strategies.

## Post-processing
1. **Hash for deduping** – compute a SHA‑256 of the normalised proposition to avoid storing the same idea twice.
2. **Attach metadata** such as source ID, position, importance and veracity scores.
3. **Embed** the idea and assign it to the nearest centroid.
4. **Update the centroid** using your EMA rule and merge metadata from prior evidence.

This ongoing centroid update process is what allows a prototype to gradually "learn" from accumulated evidence—a key distinction from static retrieval systems.
This mirrors the hippocampus→cortex flow: raw episode → event boundary → gist proposition → integrated belief.

## Reference pipeline
```python
for para in paragraphs(doc):
    ideas = agentic_splitter(para)       # TextTiling → LLM if long
    for idea in ideas:
        if is_duplicate(idea):
            continue
        meta = build_meta(doc, idea)
        vec = embed(idea)
        cid = assign_centroid(vec)
        reconcile(vec, meta, cid)
```
Latency benchmarks on a single CPU (10k ideas/min) come from using MiniLM embeddings and TextTiling first, calling the LLM only for segments over 512 tokens.

## Production checklist
- Tune max tokens per idea (60–120 tokens works well for MiniLM retrieval).
- Keep overlap small (≤20%) to avoid duplicate centroids.
- Evaluate retrieval F1 versus chunk granularity; too coarse usually hurts long‑tail recall, too fine inflates the index.
- Monitor centroid drift and auto-split if intra-distance exceeds δ.

## Pipeline strategies

For more complex workflows a `PipelineCompressionStrategy` can chain multiple
strategies. The output of one step feeds into the next, enabling filters and
summarizers to be composed.

Example configuration:

```yaml
strategy_name: pipeline
strategies:
  - strategy_name: importance
  - strategy_name: learned_summarizer
```

## Optional strategy plugins

Some strategies are distributed separately as installable packages. For example,
the `rationale_episode` strategy provides rationale-enhanced episodic memory and
related CLI tools. Install it via:

```bash
pip install compact_memory_rationale_episode_strategy
```

Once installed, its commands are available through the main `compact-memory` CLI.
