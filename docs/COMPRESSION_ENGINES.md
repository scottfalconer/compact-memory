# Compression Engines (Conceptual Overview)

This document provides a conceptual overview of various approaches to text and memory compression that can be relevant to Large Language Models (LLMs) and conversational AI systems. It explores different strategies, their trade-offs, and potential use cases.

**Note on Framework Implementation:** The concepts discussed here represent a range of ideas and techniques. For details on how to implement a compression engine within the Compact Memory framework, including the `BaseCompressionEngine` class, persistence, and integration with the CLI, please refer to the [Developing Compression Engines](./DEVELOPING_COMPRESSION_ENGINES.md) guide.

## 1. What is "Compression" in this Context?

In the context of LLMs and memory systems, "compression" refers to techniques that reduce the size or complexity of textual information while attempting to preserve its essential meaning or utility for a specific task. This is crucial for:

*   **Managing Context Window Limits:** LLMs have finite context windows. Compression helps fit more information or longer histories into this window.
*   **Reducing Computational Cost:** Processing shorter texts can be faster and cheaper.
*   **Improving Signal-to-Noise Ratio:** Removing redundant or irrelevant information can help the LLM focus on what's important.
*   **Long-Term Memory Simulation:** Storing compressed representations of past interactions or knowledge.

Compression is not necessarily about achieving the highest possible data compression ratio in a traditional sense (like Gzip). Instead, it's about "semantic compression" or "information distillation."

**Note on LLM-Dependent Engines:** Some advanced compression engines, particularly those performing abstractive summarization or complex transformations (e.g., a future `readagent_gist` engine), may require a Large Language Model (LLM) to function. Configuring these LLMs (selecting the provider, model, API keys, etc.) is crucial for such engines. For details on how to configure LLMs when using these engines via the CLI, please refer to the `compact-memory compress --help` output, the [CLI Reference](./cli_reference.md), and the [Configuration Guide](./configuration.md#llm-model-configurations-llm_models_configyaml).

## 2. Types of Compression Engines/Strategies

### a. Extractive Summarization (Truncation-based)

*   **Concept:** Selects and concatenates the most important sentences or phrases from the original text. This can be as simple as taking the first N sentences/tokens, or more sophisticated methods involving sentence scoring (e.g., based on TF-IDF, LexRank, or transformer-based sentence embeddings).
*   **Engine Examples:**
    *   `NoCompressionEngine` (with a small budget): Effectively acts like truncation.
    *   `FirstLastKEngine`: Keeps the first K and last K elements (e.g., sentences or tokens).
*   **Pros:** Simple, fast, preserves original wording.
*   **Cons:** May miss nuances if important information is not in the selected parts. Can be disjointed.
*   **Use Cases:** Compressing recent conversation history where the beginning and end are often most relevant, quick previews of documents.

### b. Abstractive Summarization

*   **Concept:** Generates new text that captures the essence of the original content. This typically involves using a separate LLM fine-tuned for summarization.
*   **Engine Examples:**
    *   An engine that internally calls a summarization model (e.g., T5, BART, or a proprietary API).
*   **Pros:** Can be more coherent and concise than extractive methods. Can synthesize information.
*   **Cons:** Computationally more expensive. Prone to hallucinations or misinterpretations by the summarization model. May lose specific details.
*   **Use Cases:** Creating summaries of documents, summarizing past conversation segments for long-term memory.

### c. Embedding-based Retrieval (Query-Relevant Compression)

*   **Concept:** Instead of a generic summary, retrieve only the chunks of text from a larger corpus that are most relevant to a current query or context. The "compression" comes from only selecting a small, relevant subset.
*   **Engine Examples:**
    *   `PrototypeEngine` (when used for querying): Retrieves relevant prototypes and associated memories, which can be seen as a form of query-specific compression of its knowledge.
    *   Any RAG (Retrieval Augmented Generation) system.
*   **Pros:** Highly relevant to the current task. Can draw from vast knowledge.
*   **Cons:** Depends heavily on the quality of embeddings and the retrieval mechanism. May not provide a good general "summary" if no specific query is present.
*   **Use Cases:** Answering questions based on a knowledge base, providing context for LLM responses.

### d. Fine-tuning / Distillation for Specific Tasks

*   **Concept:** Train a smaller model (or fine-tune an existing one) to perform a specific task that a larger model would typically do with more extensive context. The "compression" is in the model itself, which has learned to implicitly store and process information relevant to its task.
*   **Pros:** Can be very efficient at runtime for the specific task.
*   **Cons:** Requires significant effort to train and maintain. Less flexible than general-purpose compression.
*   **Use Cases:** Specialized chatbots, classifiers, information extraction tools.

### e. Delta / Change-Based Compression

*   **Concept:** For sequential data (like conversation turns or document versions), store only the differences or updates from the previous state.
*   **Pros:** Can be very efficient if changes are small.
*   **Cons:** Reconstruction can be slow if many deltas need to be applied. Not suitable for all types of text.
*   **Use Cases:** Version control, tracking changes in dynamic information.

### f. Concept-Based Compression (Knowledge Graphs / Ontologies)

*   **Concept:** Represent text as a set of entities and relationships, possibly mapped to a formal ontology or knowledge graph. The "compression" is that the structured representation is often more compact and machine-understandable than raw text.
*   **Engine Examples:**
    *   Systems that extract RDF triples or build knowledge graphs from text.
    *   `PrototypeEngine`'s prototypes can be seen as a light form of conceptual representation.
*   **Pros:** Enables complex reasoning and querying. Can link disparate pieces of information.
*   **Cons:** Complex to implement. Knowledge extraction can be error-prone.
*   **Use Cases:** Building comprehensive knowledge bases, semantic search, advanced reasoning.

## 3. Key Considerations for Choosing/Designing a Compression Engine

*   **Information Fidelity:** How much of the original, important information is preserved?
*   **Computational Cost:** How much processing power and time does compression/decompression take?
*   **Output Coherence:** Is the compressed output readable and understandable (if human-facing)?
*   **Task Relevance:** How well does the compressed output serve the intended downstream task (e.g., LLM prompting, information retrieval)?
*   **Lossiness:** Is it acceptable to lose some information, or must the compression be lossless (or near-lossless for key details)?
*   **Interpretability:** Is it important to understand how the compression decision was made?

## 4. Interaction with Compact Memory Framework

The Compact Memory library aims to provide tools and a framework (`BaseCompressionEngine`) for implementing and experimenting with various compression engines. Developers can create subclasses to implement specific strategies, manage their persistence, and integrate them into workflows or CLI operations.

For concrete examples and guidelines on building your own engine, see [Developing Compression Engines](./DEVELOPING_COMPRESSION_ENGINES.md).
