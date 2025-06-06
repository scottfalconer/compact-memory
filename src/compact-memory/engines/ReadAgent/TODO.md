# ReadAgent Engine TODO

This file tracks missing functionalities or desirable features in the base `compact-memory` project
that would benefit the `ReadAgentGistEngine`.

- **Local LLM Abstraction:** A standardized way to interact with local LLMs (e.g., via `compact_memory.local_llm` or similar) would be beneficial. Currently, the engine expects a `local_llm_pipeline` to be passed in, but a more integrated solution for model loading and inference would simplify configuration.
- **Advanced Text Splitting/Pagination Utilities:** While the engine will implement its own episode pagination, more sophisticated text segmentation utilities in `compact_memory.chunker` (e.g., LLM-aided semantic chunking) could be leveraged in the future.
- **Standardized Prompt Management:** A system for managing and versioning prompt templates within `compact-memory` could be useful for engines that rely heavily on specific prompt structures.
- **Custom `ingest` for Advanced Pagination:** If the episode pagination logic (e.g., future LLM-based pagination) becomes significantly different from what standard `Chunker` objects provide, a full override of the `ingest` method might be necessary to incorporate `_paginate_episodes` directly into the ingestion flow, instead of relying on `self.chunker` to produce episodes. For now, `_compress_chunk` assumes `self.chunker`'s output maps to episodes.
