# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Introduced `BaseVectorStore` interface to allow pluggable vector storage backends.
- Added `InMemoryVectorStore` as the default vector store, utilizing FAISS internally for similarity search.
- Added `ChromaVectorStoreAdapter` for using ChromaDB as a vector store.
- Added `FaissVectorStoreAdapter` for using a standalone FAISS index with persistence capabilities.
- Introduced `EmbeddingFunction` protocol in the embedding pipeline, allowing users to provide custom embedding functions.
- `Agent` and `ActiveMemoryManager` now accept `vector_store` and `embedding_fn` / `embedding_dim` in their constructors.
- CLI (`compact-memory agent init`, `query`, `ingest`, etc.) updated with:
    - `--vector-store-type` (memory, chroma, faiss)
    - `--vector-store-path` (for persistent stores)
    - `--vector-store-collection` (for Chroma)
    - `--embedding-provider-type` (huggingface, mock)
    - `--embedding-model-name`, `--embedding-device`, `--embedding-dim`
- Agent persistence (`agent_config.json`) now stores configurations for the selected vector store and embedding provider.
- `Agent.load_agent` class method now requires `vector_store_instance` (pre-initialized) and `embedding_fn` (if custom) to be passed, enabling flexible instantiation by the caller (e.g., the CLI).
- Created `compact_memory/embedding_providers/huggingface.py` to encapsulate Hugging Face SentenceTransformer logic, making `sentence-transformers` and `torch` optional dependencies for the core library.

### Changed
- Refactored `Agent`, `ActiveMemoryManager`, and `PrototypeSystemStrategy` to use the `BaseVectorStore` interface for all vector operations.
- `PrototypeSystemStrategy` now manages `BeliefPrototype` and `RawMemory` objects in local dictionaries instead of relying on the vector store to hold these Pydantic models directly. It includes `save_state` and `load_state` for these dictionaries.
- `embedding_pipeline.embed_text` now optionally accepts an `embedding_fn`. If not provided, it defaults to the Hugging Face implementation (`embed_text_hf`).
- `embedding_pipeline.get_embedding_dim` now primarily serves to get the dimension of the default Hugging Face model. Users of custom embedding functions must provide `embedding_dim` explicitly.
- `Agent.save_agent` and `Agent.load_agent` methods updated to handle the new vector store and embedding configurations.

### Removed
- Removed previous `JsonNpyVectorStore` and its direct filesystem persistence for vectors (assuming this was the prior primary method; this refactoring supersedes it).
- Removed direct FAISS management logic from `PrototypeSystemStrategy` and `Agent`; this is now encapsulated in `InMemoryVectorStore` or `FaissVectorStoreAdapter`.
- `BeliefPrototype.vector_row_index` field (pending confirmation in next step, but planned for removal as it's likely unused by new vector store adapters).

### Fixed
- Potential circular import for `embed_text` between `agent.py` and `prototype_system_strategy.py` by having `PrototypeSystemStrategy` import directly from `embedding_pipeline`.
