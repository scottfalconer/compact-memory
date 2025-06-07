# Compact Memory Documentation

Welcome to the detailed documentation for Compact Memory! This section provides in-depth information beyond the main project [README.md](../README.md). Whether you're looking to understand the architecture or develop new engines, these documents are here to guide you.

## Table of Contents & Document Overview

Below is a list of documents in this directory, along with a brief description of their content and intended audience.

*   **[`ARCHITECTURE.md`](./ARCHITECTURE.md)**
    *   **Content:** A detailed explanation of the Compact Memory platform's architecture, including core modules, data models, ingestion flow, querying mechanisms, and testing engines.
    *   **Audience:** Developers contributing to the Compact Memory core, or those seeking a deep understanding of its internal workings.


*   **Conceptual Guides for Engines**
    *   **[`COMPRESSION_ENGINES.md`](./COMPRESSION_ENGINES.md)**: Discusses techniques for segmenting source documents and post-processing for creating effective compression engines. Focuses on the "what" and "why."
        *   **Audience:** Researchers and developers designing new compression engines.
*   **[`QUERY_TIPS.md`](./QUERY_TIPS.md)**: Explains how to shape search queries for effective retrieval, especially when using structured note layouts.
*   **[`MIGRATION_TO_ENGINES.md`](./MIGRATION_TO_ENGINES.md)**: Describes how to update existing code from strategies and containers to the new engine API.
        *   **Audience:** Users and developers working with the querying aspects of Compact Memory.
    *   **[`EXPLAINABLE_COMPRESSION.md`](./EXPLAINABLE_COMPRESSION.md)**: (Assuming content based on title) Discusses approaches to make compression engines more transparent and understandable.
        *   **Audience:** Researchers and developers interested in the interpretability of memory systems.

*   **Developer Guides**
    *   **[`ENGINE_DEVELOPMENT.md`](./ENGINE_DEVELOPMENT.md)**: Provides practical guidance and steps for implementing new `BaseCompressionEngine` implementations within the Compact Memory framework. Focuses on the "how-to."
        *   **Audience:** Developers actively building new compression engines.
    *   **[`DEVELOPING_COMPRESSION_ENGINES.md`](./DEVELOPING_COMPRESSION_ENGINES.md)**: Best practices for writing, testing, and sharing `BaseCompressionEngine` implementations.
        *   **Audience:** Contributors preparing reusable compression engines.
    *   **[`DEVELOPING_VALIDATION_METRICS.md`](./DEVELOPING_VALIDATION_METRICS.md)**: Offers guidance on creating custom `ValidationMetric` classes to evaluate the performance of compression engines.
        *   **Audience:** Developers and researchers looking to implement new ways of measuring memory effectiveness.
    *   **[`STORAGE_FORMAT.md`](./STORAGE_FORMAT.md)**: Describes an example on-disk format previously used for Compact Memory.
        *   **Audience:** Developers needing to understand or interact with the persistence layer.

*   **[`SHARING_ENGINES.md`](./SHARING_ENGINES.md)**: Guidelines for packaging and sharing custom-developed compression engines.
    *   **Audience:** Developers and researchers who want to contribute or distribute their work.

*   **Preprocessing & Cleanup**
    *   **Content:** Compact Memory does not perform built-in line filtering. Use the `preprocess_fn` hook when constructing an `Agent` or engine to clean or normalize text.
        Example utilities include spaCy pipelines, regex removal, or custom LLM summarizers.
    *   **Audience:** Developers needing fine-grained control over input text before compression.

## Suggested Reading Paths

*   **For Users Wanting to Apply Compact Memory:**
    1.  Start with the main project [README.md](../README.md) and the [USAGE guide](./USAGE.md).
    2.  Review [`QUERY_TIPS.md`](./QUERY_TIPS.md) for effective information retrieval.

*   **For Developers Building New Compression Engines:**
    1.  Study the conceptual approaches in [`COMPRESSION_ENGINES.md`](./COMPRESSION_ENGINES.md).
    2.  Follow the practical implementation guide in [`ENGINE_DEVELOPMENT.md`](./ENGINE_DEVELOPMENT.md).
    3.  Refer to [`ARCHITECTURE.md`](./ARCHITECTURE.md) for how engines fit into the broader system.
    4.  Learn about evaluation with [`DEVELOPING_VALIDATION_METRICS.md`](./DEVELOPING_VALIDATION_METRICS.md).

*   **For Those Contributing to the Core Platform:**
    1.  Begin with [`ARCHITECTURE.md`](./ARCHITECTURE.md).
    2.  Review relevant developer guides based on the area of contribution.

We encourage you to explore these documents to get the most out of Compact Memory. If you find areas for improvement or clarification, please feel free to open an issue or suggest changes!
