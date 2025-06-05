# Compact Memory Architecture

This document provides a detailed technical explanation of the Compact Memory platform's architecture, intended for developers contributing to the core or those seeking a deep understanding of its internal workings.

## Overview

`compact-memory` implements a coarse prototype memory system for storing and
retrieving textual information. Incoming text is converted into vector
embeddings and "snap-assigned" to belief prototypes which act as stable
centroids. The codebase is organised into a library providing Python APIs and
a CLI (`compact-memory`).

The design follows the hypotheses documented in `PROJECT_VISION.md`:
prototypes reduce storage and search cost while providing more robust
gist-based reasoning. The implementation emphasises pluggability so
alternative memory creation, embedding and storage mechanisms can be
experimented with.

## Package layout

```
├── compact_memory/           # main library
├── CompressionEngine/        # core compression engine logic and specific engine implementations
├── examples/              # onboarding demo
├── tests/                 # unit tests using mock embeddings
└── docs/                  # documentation
```

### Core modules
<!-- SUGGESTION: A diagram illustrating the interaction between core modules (e.g., CompressionEngine, embedding_pipeline.py, chunker.py, etc.) could be helpful here. -->

- **`CompressionEngine/core/engines_abc.py` & implementations**: Defines the `CompressionEngine` interface and specific engine implementations (e.g., `PrototypeEngine`, `SummarizationEngine`). Engines are responsible for the core logic of text compression, and some may handle aspects of ingestion, querying, and stateful memory management (like prototype updates) if they are designed for long-term memory.
- **`embedding_pipeline.py`** – loads a SentenceTransformer model and
  exposes `embed_text`. A deterministic `MockEncoder` is available for
  tests. Embeddings are cached and normalised.
- **`CompressionEngine/core/pipeline_engine.py`** – implements `PipelineCompressionEngine`
  allowing multiple compression steps/engines to be chained.
- **`chunker.py`** – implements sentence-window based chunking with token
  overlap and a fixed-size fallback. The registry allows different
  chunkers to be plugged in via config. Engines may use these chunkers.
- **`memory_creation.py`** – small utilities to create "memory" texts
  from raw documents. Includes identity, extractive, fixed chunk and
  LLM-driven variants so experiments can measure which produces better
  prototypes (often used in conjunction with prototype-based engines).
- **`cli.py`** – Typer-based command line app supporting `init`,
  `stats`, `validate`, `clear`, `download-model`, `download-chat-model`,
  `experiment` and `dev inspect-engine`. Persistence (if applicable to an engine) is
  locked during writes to avoid corruption.
- Additional vector store implementations can be developed by
  extending the interfaces used by a vector store interface. This keeps
  engines that use vector stores decoupled from any particular storage backend or embedding
  provider.

### Data models
<!-- SUGGESTION: A simple diagram showing the relationship between BeliefPrototype and RawMemory data models would be useful. -->

`models.py` defines two `pydantic` models:

- `BeliefPrototype` – metadata about a prototype without the vector.
  Tracks strength, confidence, timestamps and the IDs of constituent
  memories.
- `RawMemory` – a chunk of source text with optional embedding and the
  prototype it belongs to.

They are stored in JSON and referenced by ID in the NPY vector arrays.

## Ingestion flow
<!-- SUGGESTION: A diagram illustrating the ingestion flow (Chunking -> Embedding -> Prototype Search -> Updating) would be beneficial here. -->

1. **Chunking** – text is split into sentence windows using
   `SentenceWindowChunker` or into belief-sized ideas with
   `AgenticChunker` (both are registered chunkers). This step is often orchestrated by or within a `CompressionEngine`.
2. **Embedding** – each chunk is converted into a normalised vector via
   `embed_text`.
3. **Prototype search (for prototype-based engines)** – the store finds the nearest prototype by
   cosine similarity. If no similarity exceeds the threshold, the engine (e.g., `PrototypeEngine`)
   spawns a new prototype.
4. **Updating (for stateful engines)** – existing prototypes might be updated using an exponential
   moving average. New memories are appended to storage and
   evidence logged as defined by the specific engine.

Engines or related components may keep an LRU cache of recent SHA‑256 hashes to skip duplicates
quickly.

## Querying
<!-- SUGGESTION: A diagram illustrating the querying process (Query Embedding -> Prototype Retrieval -> Memory Scoring) would be beneficial here. -->

For engines that support querying (e.g., prototype-based engines):
1. The query text is embedded and the top `k` prototypes are retrieved.
2. Constituent memories from those prototypes are scored by similarity to
   the query embedding.
3. The results include the ranked prototypes and the top memories.

## CLI

`compact-memory` is implemented using Typer and exposes subcommands for
initialising memory stores (if applicable for certain engines), inspecting stored prototypes (for relevant engines), and running
experiments. The CLI is the primary interface, and a Colab notebook will
provide a graphical option in the future.

## Testing

The test suite uses the deterministic `MockEncoder` to avoid heavy model
loads. `pytest` exercises engine logic, CLI commands, chunkers, the
embedding pipeline and vector stores. Continuous integration ensures
that persistence round‑trips (where applicable) and query ranking work as expected.

## Rationale

The current implementation aims to validate hypotheses like coarse prototype memory
with a lightweight, easily inspectable codebase.
Abstractions for embedding models and vector stores allow the
system to evolve without rewriting engine logic. Unit tests with a
mock encoder keep the feedback loop fast and deterministic.

