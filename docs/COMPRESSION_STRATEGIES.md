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
