# Compact Memory Documentation

Welcome to the detailed documentation for Compact Memory! This section provides in-depth information beyond the main project [README.md](../README.md). Whether you're looking to understand the architecture, develop new strategies, or run comprehensive experiments, these documents are here to guide you.

## Table of Contents & Document Overview

Below is a list of documents in this directory, along with a brief description of their content and intended audience.

*   **[`ARCHITECTURE.md`](./ARCHITECTURE.md)**
    *   **Content:** A detailed explanation of the Compact Memory platform's architecture, including core modules, data models, ingestion flow, querying mechanisms, and testing strategies.
    *   **Audience:** Developers contributing to the Compact Memory core, or those seeking a deep understanding of its internal workings.

*   **[`PROJECT_VISION.md`](./PROJECT_VISION.md)**
    *   **Content:** A conceptual document outlining the vision behind Compact Memory. It discusses illustrative memory management strategies (like the Prototype System and ActiveMemoryManager), prompt assembly, and key areas for experimentation and learning.
    *   **Audience:** Researchers, developers looking for inspiration for new strategies, and anyone interested in the theoretical underpinnings of the platform.

*   **Conceptual Guides for Strategies**
    *   **[`COMPRESSION_STRATEGIES.md`](./COMPRESSION_STRATEGIES.md)**: Discusses techniques for segmenting source documents and post-processing for creating effective compression strategies. Focuses on the "what" and "why."
        *   **Audience:** Researchers and developers designing new compression strategies.
    *   **[`QUERY_TIPS.md`](./QUERY_TIPS.md)**: Explains how to shape search queries for effective retrieval, especially when using structured note layouts.
        *   **Audience:** Users and developers working with the querying aspects of Compact Memory.
    *   **[`EXPLAINABLE_COMPRESSION.md`](./EXPLAINABLE_COMPRESSION.md)**: (Assuming content based on title) Discusses approaches to make compression strategies more transparent and understandable.
        *   **Audience:** Researchers and developers interested in the interpretability of memory systems.

*   **Developer Guides**
    *   **[`DEVELOPING_COMPRESSION_STRATEGIES.md`](./DEVELOPING_COMPRESSION_STRATEGIES.md)**: Provides practical guidance and steps for implementing new `CompressionStrategy` modules within the Compact Memory framework. Focuses on the "how-to."
        *   **Audience:** Developers actively building new compression strategies.
    *   **[`DEVELOPING_VALIDATION_METRICS.md`](./DEVELOPING_VALIDATION_METRICS.md)**: Offers guidance on creating custom `ValidationMetric` classes to evaluate the performance of compression strategies.
        *   **Audience:** Developers and researchers looking to implement new ways of measuring memory effectiveness.
    *   **[`STORAGE_FORMAT.md`](./STORAGE_FORMAT.md)**: Describes an example on-disk format previously used for Compact Memory.
        *   **Audience:** Developers needing to understand or interact with the persistence layer.

*   **Running & Managing Experiments**
    *   **[`RUNNING_EXPERIMENTS.md`](./RUNNING_EXPERIMENTS.md)**: A guide to using the experimentation framework, including setting up experiments, running them via the CLI, and understanding the output.
        *   **Audience:** Researchers and developers evaluating and comparing compression strategies.
    *   **[`ADVANCED_PARAMETER_TUNING.md`](./ADVANCED_PARAMETER_TUNING.md)**: (Assuming content based on title) Covers more advanced techniques for tuning parameters of compression strategies and the experimentation setup.
        *   **Audience:** Experienced users looking to optimize performance.
    *   **[`SHARING_STRATEGIES.md`](./SHARING_STRATEGIES.md)**: (Assuming content based on title) Guidelines or methods for packaging and sharing custom-developed compression strategies.
        *   **Audience:** Developers and researchers who want to contribute or distribute their work.

*   **Preprocessing & Cleanup**
    *   **Content:** Compact Memory does not perform built-in line filtering. Use the `preprocess_fn` hook when constructing an `Agent` or strategy to clean or normalize text.
        Example utilities include spaCy pipelines, regex removal, or custom LLM summarizers.
    *   **Audience:** Developers needing fine-grained control over input text before compression.

## Suggested Reading Paths

*   **For Users Wanting to Apply Compact Memory:**
    1.  Start with the main project [README.md](../README.md) for installation and basic usage.
    2.  Review [`QUERY_TIPS.md`](./QUERY_TIPS.md) for effective information retrieval.
    3.  If running extensive evaluations, consult [`RUNNING_EXPERIMENTS.md`](./RUNNING_EXPERIMENTS.md).

*   **For Developers Building New Compression Strategies:**
    1.  Understand the core concepts in [`PROJECT_VISION.md`](./PROJECT_VISION.md).
    2.  Study the conceptual approaches in [`COMPRESSION_STRATEGIES.md`](./COMPRESSION_STRATEGIES.md).
    3.  Follow the practical implementation guide in [`DEVELOPING_COMPRESSION_STRATEGIES.md`](./DEVELOPING_COMPRESSION_STRATEGIES.md).
    4.  Refer to [`ARCHITECTURE.md`](./ARCHITECTURE.md) for how strategies fit into the broader system.
    5.  Learn about evaluation with [`DEVELOPING_VALIDATION_METRICS.md`](./DEVELOPING_VALIDATION_METRICS.md) and [`RUNNING_EXPERIMENTS.md`](./RUNNING_EXPERIMENTS.md).

*   **For Those Contributing to the Core Platform:**
    1.  Begin with [`ARCHITECTURE.md`](./ARCHITECTURE.md).
    2.  Understand the vision in [`PROJECT_VISION.md`](./PROJECT_VISION.md).
    3.  Review relevant developer guides based on the area of contribution.

We encourage you to explore these documents to get the most out of Compact Memory. If you find areas for improvement or clarification, please feel free to open an issue or suggest changes!
