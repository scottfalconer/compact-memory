# Core Memory Library Enhancement TODOs (for NeocortexTransfer & Similar Engines)

Based on the implementation of the `NeocortexTransfer` engine, the following features or enhancements to a hypothetical `core-memory` library could significantly simplify the development of cognitively-inspired memory engines:

1.  **Standardized Knowledge Base/Graph Interface:**
    *   **Description:** Provide a built-in or plugin-friendly interface for managing and querying a structured knowledge base or semantic graph. This would allow engines to store, link, and retrieve factual data, concepts, and their relationships more formally than simple dictionaries.
    *   **Benefit for `NeocortexTransfer`:** Would improve the "Prior Knowledge" aspect of Semantic Comprehension (US1) and the "Integration" part of Consolidation (US4).

2.  **Advanced Text Processing Utilities:**
    *   **Description:** Integrate or provide easy access to more sophisticated text processing tools beyond basic tokenization. This could include:
        *   Sentence boundary detection.
        *   Part-of-speech tagging.
        *   Named entity recognition (NER).
        *   Coreference resolution.
        *   Text summarization (for gists/main ideas).
    *   **Benefit for `NeocortexTransfer`:** Would greatly enhance the "Semantic Parsing" in US1, allowing for richer understanding and more meaningful "chunks" in US2.

3.  **Formal Working Memory Abstraction:**
    *   **Description:** A dedicated `WorkingMemory` class or module that handles:
        *   Configurable capacity limits.
        *   Recency-weighted item access/eviction.
        *   Optional item activation levels that decay over time (simulating fading memory).
        *   Mechanisms for "refreshing" items (boosting activation).
    *   **Benefit for `NeocortexTransfer`:** Would make `self.working_memory_context` in US1 and STM simulations in US2 more robust and feature-rich.

4.  **Event/Hook System for Cognitive State Changes:**
    *   **Description:** A system allowing engines to emit events or trigger hooks at key points in their processing pipeline (e.g., `on_comprehension_error`, `on_novel_information_detected`, `on_memory_consolidated`, `on_retrieval_failure`). Other components or plugins could subscribe to these.
    *   **Benefit for `NeocortexTransfer`:** Would formalize "Comprehension Monitoring" (US1) and allow for more modular reactions to internal state changes (e.g., triggering specific learning strategies).

5.  **Memory Consolidation Primitives:**
    *   **Description:** Provide optional, configurable utilities for common consolidation tasks:
        *   Scheduled background processing for consolidation routines.
        *   Decay functions for memory attributes (e.g., strength, activation) over time.
        *   Mechanisms for strengthening traces based on retrieval or salience.
        *   Basic trace linking or merging strategies.
    *   **Benefit for `NeocortexTransfer`:** Could simplify parts of the `_consolidate_and_integrate` logic in US4, especially if background processing or standardized decay/strengthening models are desired.

6.  **Extensible Retrieval Strategy Framework:**
    *   **Description:** Allow different retrieval algorithms to be plugged into the memory system. Beyond simple keyword matching, this could include:
        *   Semantic similarity search (e.g., using vector embeddings).
        *   Graph-based traversal for finding related memories.
        *   Context-aware retrieval that considers the current state of working memory.
    *   **Benefit for `NeocortexTransfer`:** Would make `_retrieve_and_reintegrate` in US5 more powerful and flexible, allowing for more nuanced cueing and recall.

7.  **Standardized Metacognitive Data Handling:**
    *   **Description:** Define a common way to attach, store, and process metacognitive information associated with memory traces, such as:
        *   Confidence scores.
        *   "Feeling of knowing" indicators.
        *   Source monitoring information (how/where was this learned?).
        *   Uncertainty flags.
    *   **Benefit for `NeocortexTransfer`:** Would standardize how confidence is calculated and used in US5 and provide a richer substrate for self-monitoring.

8.  **Trace Lifecycle Management:**
    *   **Description:** Helpers for managing the lifecycle of memory traces, including explicit states (e.g., `new`, `learning`, `consolidating`, `stable`, `decaying`, `forgotten`) and transitions between them.
    *   **Benefit for `NeocortexTransfer`:** Would provide a more formal structure for the `status` and `consolidation_level` attributes of traces.

These suggestions aim to provide developers with more powerful building blocks, reducing the amount of foundational cognitive simulation they need to implement from scratch for each new engine.
