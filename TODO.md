# TODO

This file tracks outstanding work based on `AGENTS.md`.

## Prototype memory system
 - Update prototypes via EMA and add health checks for splitting or merging.
- Implement memory/prototype decay and conflict resolution flows.

## Retrieval pipeline
- Add two-tier retrieval with fine re-ranking or raw memory lookup.
- Provide decoding of prototypes into human-readable summaries.

## Memory creation and embeddings
- Support pluggable memory creation engines (LLM summary, chunking, extractive).
- Replace random embeddings with real local or remote models.
- Allow locally runnable models as an option.

## Research and evaluation
- Benchmark prototype-based retrieval vs raw memory retrieval.
- Determine effective τ adaptation heuristics.
- Compare memory_text representation strategies.
- Evaluate methods for decoding prototype meaning.

## CLI and usability
- Expose configuration options (threshold, embedding model, memory creator).
- Allow configuration of adaptive threshold parameters.
- Improve documentation and installation instructions.

