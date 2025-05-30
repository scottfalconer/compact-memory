**I. Vision: The Cognitively-Inspired Gist Memory Agent**

* **Core Mission:** To develop an agent that intelligently captures, stores, and utilizes the "gist" – the essential, abstract meaning – of information, combating information overload and enabling more effective knowledge utilization.
* **Guiding Philosophy:** We draw inspiration from human cognitive processes, including memory, attention, and learning. Our aim is to create mechanisms that are more organic and adaptive than purely hardcoded logic. We believe this approach will lead to a more robust, flexible, and insightful agent.
* **Development Tenet: Design for Experimentation:** A fundamental principle of this project is that components modeling cognitive functions must be designed with tunable parameters. Our experimentation framework is crucial for validating hypotheses about these mechanisms, optimizing their performance, and driving iterative improvement.

**II. Core Cognitive Analogs & System Architecture**

Our system architecture distinguishes between long-term knowledge storage and a dynamic, active memory for current interactions.

**A. Long-Term Memory (LTM): The Prototype System – Capturing the Gist**

* **Concept:** Analogous to human semantic and episodic long-term memory, this system stores consolidated knowledge.
* **Mechanism: Coarse Prototype Memory System**
    * **Gist-Based Processing:** Inspired by Fuzzy-Trace Theory, we prioritize the extraction and storage of the essential meaning ("gist") over verbatim details. This promotes durable and flexible knowledge.
    * **Prototypes as Conceptual Centroids:** Incoming information (memories) are snap-assigned to the nearest existing "prototype" (a vector representing a conceptual gist) or spawn new prototypes if sufficiently novel.
    * **Prototype Evolution:** Prototypes are dynamic. Their vector representations and textual summaries evolve via an Exponential Moving Average (EMA) as new, related memories are assigned, reflecting how concepts can refine over time. Their `strength` increases with supporting evidence.
    * **Schema-Driven Assimilation:** New information is integrated by relating it to these existing conceptual structures (prototypes), mirroring how human understanding is often guided by existing schemas.
* **Key Tunable Parameters for LTM (via Experimentation Framework):**
    * `similarity_threshold` (τ): For assigning memories to prototypes or spawning new ones.
    * `ema_alpha` (α): Learning rate for prototype vector/summary updates.
    * Prototype health metrics & thresholds: For potential future splitting/merging of prototypes.
    * Parameters for `MemoryCreator` strategies (e.g., chunk size, summarization detail).

**B. Active/Working Memory: Dynamic Conversational Context – Thinking in the Moment**

* **Concept:** Simulates human active/working memory, providing a limited-capacity buffer that holds and processes information relevant to the current interaction, especially for the `talk` command's LLM. This is managed by the `ActiveMemoryManager`.
* **Core Characteristics (Inspired by Human Cognition):**
    1.  **Limited Capacity:** The total information assembled for the LLM prompt strictly adheres to the model's token limits.
    2.  **Dynamic Content:** The content is not a fixed window but is dynamically selected and prioritized based on several factors.
    3.  **Recency (Activation Decay):** Recently processed conversational turns have a higher baseline "activation level," making them readily available. This activation naturally decays over time or with new inputs, unless refreshed.
    4.  **Trace Strength (Intrinsic Importance):** Each conversational turn acquires a `trace_strength` score. This reflects its inherent significance, calculated based on factors like:
        * Semantic novelty at the time of occurrence.
        * Presence of salient entities (e.g., identified by spaCy NER).
        * Degree to which it activated or led to updates in LTM (prototypes).
        * (Future) Explicit user feedback (e.g., "this is important").
        This `trace_strength` makes a turn more resistant to being pruned from the overall history buffer.
    5.  **Current Activation Level (Contextual Relevance & Connection):** A turn's `trace_strength` is modulated by its `current_activation_level`. This dynamic level is boosted when a historical turn is semantically similar (connected) to the *current user query* or the immediate conversational context. This models how current cues "light up" relevant past information (spreading activation).
* **Mechanism (`ActiveMemoryManager`):**
    * Stores `ConversationalTurn` objects, each enriched with its text, embedding, `trace_strength`, and dynamically updated `current_activation_level`.
    * Implements algorithms for calculating `trace_strength` upon turn creation.
    * Manages `current_activation_level` through decay and relevance-based boosting.
    * Employs **Prioritized Pruning** for the main history buffer: When the buffer exceeds its max size, it prunes turns with the lowest combined score of recency, activation, and trace strength.
* **Key Tunable "Meta-Parameters" for Active Memory (via Experimentation Framework):**
    * Weights for factors contributing to `trace_strength` (e.g., `config_weight_novelty`, `config_weight_salient_entities`).
    * Parameters for `current_activation_level` dynamics (e.g., `config_initial_activation`, `config_activation_decay_rate`, `config_relevance_boost_factor`).
    * History buffer management (e.g., `config_max_history_buffer_turns`, weights for pruning decision).
    * Parameters for selecting history for the prompt (see next section).

**C. Attentional Focus & Prompt Assembly for LLM Interaction – Orchestrating Thought**

* **Concept:** The agent constructs a focused, token-limited prompt for its internal LLM by drawing from both the prioritized Active/Working Memory and selectively retrieved LTM. This simulates an "attentional spotlight."
* **Mechanism (Orchestrated by `talk` command logic using `ActiveMemoryManager` and `Agent`):**
    1.  **Prioritize Current Input:** The current user message is the primary focus.
    2.  **Select from Active Memory (STM):** The `ActiveMemoryManager` provides a selection of historical turns for the prompt. This selection (the "tree/branches" of active thought) is based on:
        * A few *forced recent* turns.
        * A limited number of older turns with high *current activation levels* (strong connection to the current query), further prioritized by their *trace strength*.
        This selection is then pruned to fit a pre-defined token budget for STM within the overall prompt.
    3.  **Retrieve Relevant Gist from LTM:** The current query (potentially augmented by context from selected STM) is used to query the `Agent`'s LTM, retrieving a small number of the most relevant prototype summaries and/or key memory snippets.
    4.  **Combine and Finalize:** The selected STM and LTM components are combined. If this combined context still exceeds the LLM's absolute input limit, `LocalChatModel.prepare_prompt()` provides a final intelligent summarization/recap.
* **Goal:** Maximize the density of relevant information (both recent interaction and long-term knowledge) within the LLM's context window.
* **Key Tunable Parameters (via Experimentation Framework):**
    * Token budget allocation ratios for STM vs. LTM within the prompt.
    * `config_prompt_num_forced_recent_turns`, `config_prompt_max_activated_older_turns`, `config_prompt_activation_threshold_for_inclusion` (for STM selection from `ActiveMemoryManager`).
    * `top_k_prototypes`, `top_k_memories` for LTM retrieval.
    * Parameters for `LocalChatModel.prepare_prompt()` (recap length, etc.).

**III. Learning & Adaptation in the Agent**

* **From New Information:** New data refines LTM prototypes (via EMA) and populates Active Memory, contributing to `trace_strength` calculation.
* **From Interaction & Feedback (Implicit & Explicit):**
    * User corrections or clarifications (even if just ingested as new memories with high `trace_strength`) will influence LTM and future STM selections.
    * (Future) Direct feedback mechanisms can more explicitly adjust `trace_strength` or trigger LTM re-evaluation.
* **Through Experimentation:** The primary mode of meta-learning and system optimization. By testing different configurations of LTM and Active Memory parameters, we refine the agent's cognitive strategies.

**IV. Guiding Principles for Developers**

1.  **Embrace Cognitive Inspiration:** When designing new features or refining existing ones, actively consider analogies from human cognition (e.g., attention, memory consolidation, contextual recall) as a rich source of design patterns.
2.  **Design for Tunability and Experimentation:** Identify key parameters in your algorithms that represent choices in how a cognitive process is modeled. Expose these so they can be systematically tested and optimized via the experimentation framework.
3.  **Prioritize Clarity of Mechanism:** While inspired by complex systems, the implemented mechanisms should be understandable, debuggable, and their impact measurable.
4.  **Modularity:** Encapsulate complex cognitive models (like the `ActiveMemoryManager` or the `PrototypeSystem`) into well-defined modules with clear interfaces.
5.  **Iterate and Validate:** Use the experimentation framework not just for tuning, but for validating that these cognitively-inspired mechanisms actually lead to improved performance and desired emergent behaviors.

**V. Key Unknowns & Areas for Experimental Validation (Update this section regularly)**

* Optimal weighting schemes for `trace_strength` factors in `ActiveMemoryManager`.
* Most effective decay rates and boosting factors for `current_activation_level`.
* Best strategies for pruning the `ActiveMemoryManager` history buffer (balancing recency, strength, activation).
* Optimal parameters for selecting STM turns for the prompt (how many recent vs. how many older-but-relevant).
* The ideal token budget allocation between STM, LTM, and current query within the final LLM prompt.
* How to effectively measure the "quality" or "human-likeness" of the agent's active memory management.
* Impact of different LTM prototype granularities (controlled by τ) on the quality of information retrieved for Active Memory.

