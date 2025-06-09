# Compact Memory Architecture

This document provides a detailed technical explanation of the Compact Memory platform's architecture, intended for developers contributing to the core or those seeking a deep understanding of its internal workings.

## Overview

`compact-memory` implements a coarse prototype memory system for storing and
retrieving textual information. Incoming text is converted into vector
embeddings and "snap-assigned" to belief prototypes which act as stable
centroids. The codebase is organised into a library providing Python APIs and
a CLI (`compact-memory`).

The design follows the core hypotheses of the project: prototypes reduce
storage and search cost while providing more robust gist-based
reasoning. The implementation emphasises pluggability so
alternative memory creation, embedding and storage mechanisms can be
experimented with.

## Package layout

```
├── compact_memory/           # main library
├── examples/              # onboarding demo
├── tests/                 # unit tests using mock embeddings
└── docs/                  # documentation
```

### Core modules
<!-- SUGGESTION: A diagram illustrating the interaction between core modules (agent.py, embedding_pipeline.py, chunker.py, etc.) could be helpful here. -->

- **`agent.py`** – orchestrates ingestion and querying using a
  a vector store interface. It handles deduplication, chunking, embedding and
  prototype updates. The default similarity threshold is 0.8.
- **`embedding_pipeline.py`** – loads a SentenceTransformer model and
  exposes `embed_text`. A deterministic `MockEncoder` is available for
  tests. Embeddings are cached and normalised.
- **`compression/pipeline_engine.py`** – implements `PipelineBaseCompressionEngine`
  allowing multiple compression steps to be chained.
- **`BaseCompressionEngine` (and its implementations, often referred to as "Compression Engines"):** These are primarily responsible for:
    - Defining how raw text is chunked (often using a `Chunker` like the one described in `chunker.py`) and how those chunks might be further compressed or transformed.
    - Orchestrating the ingestion process: This includes embedding the processed chunks, performing deduplication (maintaining a list of text content hashes), and preparing data for the vector store.
    - Managing `entries.json`: This file, saved by the engine, contains a list of dictionaries `{"id": "...", "text": "..."}` representing the ingested items. This serves as the engine's own record of what it has processed.
    - Interacting with the `VectorStore` for adding new data (ID, text, and vector tuples via `add_texts_with_ids_and_vectors`), searching for similar items (`find_nearest`), and retrieving texts for recalled IDs (`get_texts_by_ids`).
- **`VectorStore` (ABC and its implementations like `InMemoryVectorStore`, `PersistentFaissVectorStore`):** These are responsible for:
    - Storing and efficiently indexing vector embeddings.
    - Storing the associated text data for each embedding, retrievable by ID.
    - Managing their own persistence. For example, `InMemoryVectorStore` now saves its state (including embeddings, text entries, and prototype metadata) into files like `embeddings.npy`, `text_entries.json`, and `prototypes_meta.json` *within its designated `vector_store_data` directory*. `PersistentFaissVectorStore` similarly manages its Faiss index and other metadata files in its own directory. The engine no longer handles a global `embeddings.npy` file.
    - Providing a standardized interface for interaction, including methods like `count()`, `get_texts_by_ids(ids)`, and `add_texts_with_ids_and_vectors(data)`. This clear separation allows for different storage backends to be used without altering core engine logic.
- **`chunker.py`** – implements sentence-window based chunking with token
  overlap and a fixed-size fallback. The registry allows different
  chunkers to be plugged in via config.
- **`memory_creation.py`** – small utilities to create "memory" texts
  from raw documents. Includes identity, extractive, fixed chunk and
  LLM-driven variants for exploring which approach produces better
  prototypes.
- **`cli.py`** – Typer-based command line app supporting `init`,
  `stats`, `validate`, `clear`, `download-model`, `download-chat-model`,
  and `engine inspect`. Persistence is
  locked during writes to avoid corruption.

### Data models
<!-- SUGGESTION: A simple diagram showing the relationship between BeliefPrototype and RawMemory data models would be useful. -->

`models.py` defines two `pydantic` models:

- `BeliefPrototype` – metadata about a prototype without the vector.
  Tracks strength, confidence, timestamps and the IDs of constituent
  memories.
- `RawMemory` – a chunk of source text with optional embedding and the
  prototype it belongs to.

These models are used internally by vector stores. For persistence:
- The `BaseCompressionEngine` saves a file named `entries.json`, which is a simple list of dictionaries, each containing an `id` and the original `text` of an ingested item. This serves as the engine's primary record.
- `VectorStore` implementations handle their own detailed persistence. For example:
    - `InMemoryVectorStore` saves `text_entries.json` (derived from `RawMemory` objects), `prototypes_meta.json` (derived from `BeliefPrototype` objects), and `embeddings.npy` (the actual vectors) within its dedicated `vector_store_data` sub-directory.
    - `PersistentFaissVectorStore` similarly saves its own versions of these files (often with slightly different names like `memories.json`, `prototypes.json`, `vectors.npy`) along with its `index.faiss` file in its data directory.
The key change is that vector data and detailed memory/prototype metadata are now encapsulated within each vector store's own persistence directory, rather than the engine managing a global `embeddings.npy` file.

## Ingestion flow

The `BaseCompressionEngine` handles the ingestion of new text data. The typical flow is:

1.  **Chunking**: Input text is divided into smaller pieces by a `Chunker`.
2.  **Text Processing (Optional)**: Each chunk may undergo further compression or transformation by the specific engine's logic (`_compress_chunk` method).
3.  **Embedding**: The processed chunks are converted into vector embeddings.
4.  **Deduplication**: The engine checks for duplicates using a set of hashes of previously ingested content (`memory_hashes`).
5.  **Data Preparation**: For new, unique content, the engine generates a unique ID and collects tuples of `(id, processed_text, vector)`.
6.  **Storage**: The engine calls `self.vector_store.add_texts_with_ids_and_vectors(data)` to pass all new entries to the `VectorStore`. Simultaneously, the engine updates its own `self.memories` list (which is saved as `entries.json`).

## Querying

The `BaseCompressionEngine` facilitates querying the stored information:

1.  **Query Embedding**: The input query text is transformed into a vector embedding using the same embedding function used during ingestion.
2.  **Nearest Neighbor Search**: The engine calls `self.vector_store.find_nearest(query_vector, k)` to retrieve the IDs of the `k` most similar items from the `VectorStore` along with their similarity scores.
3.  **Text Retrieval**: The engine then calls `self.vector_store.get_texts_by_ids(retrieved_ids)` to fetch the corresponding text content for the retrieved IDs from the `VectorStore`.
4.  **Results**: The engine combines the IDs, scores, and retrieved texts to produce the final list of results.

## CLI

`compact-memory` is implemented using Typer and exposes subcommands for
initialising a memory store and inspecting stored prototypes. The CLI is
the primary interface, and a Colab notebook will provide a graphical
option in the future.

## Testing

The test suite uses the deterministic `MockEncoder` to avoid heavy model
loads. `pytest` exercises the agent logic, CLI commands, chunkers, the
embedding pipeline and both vector stores. Continuous integration ensures
that persistence round‑trips and query ranking work as expected.

## Rationale

The current implementation aims to validate the coarse prototype memory
hypothesis with a lightweight, easily inspectable codebase.
vectors. Abstractions for embedding models and vector stores allow the
system to evolve without rewriting the agent logic. Unit tests with a
mock encoder keep the feedback loop fast and deterministic.

