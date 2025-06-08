"""Cognitively inspired compression engine."""

from compact_memory.engines.base import BaseCompressionEngine
from .registry import register_compression_engine


class NeocortexTransfer(BaseCompressionEngine):
    """
    A compression engine that simulates human cognitive processes for
    comprehending, storing, and retrieving information.
    """

    id = "neocortex_transfer"

    def __init__(
        self, name: str = "NeocortexTransfer", version: str = "0.1.0", **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.version = version
        # Initialize any necessary attributes for User Story 1 (Semantic Comprehension)
        self.working_memory_context = []  # To maintain context
        self.prior_knowledge = {
            "example_concept": "This is a known concept."
        }  # Example prior knowledge
        self.max_context_size = 5  # Max items in working_memory_context
        self.long_term_memory_store = []  # To store LTM traces
        self.trace_id_counter = 0  # For unique trace IDs
        # TODO: Add more attributes as needed for other user stories

    def compress(self, text: str, **kwargs) -> dict:
        """
        Main method to process and compress text based on cognitive simulation.
        This will orchestrate the stages from comprehension to encoding.
        """
        # Stage 1: Semantic Comprehension
        comprehended_info = self._semantic_comprehension(text)
        tokens = text.split()
        five_word_gist = " ".join(tokens[:5]).rstrip(".,!?")

        # Stage 2: Short-Term Retention and Working Memory Management
        retained_info = self._short_term_retention(comprehended_info)

        # Stage 3: Encoding into Long-Term Memory
        encoded_info = self._encode_to_long_term_memory(retained_info)

        # Actual "compression" output is the LTM trace or a summary of it.
        return {
            "message": f"Encoded: {encoded_info.get('content')}",
            "trace_status": encoded_info.get("status"),
            "trace_strength": encoded_info.get("encoding_strength"),
            "content": five_word_gist,
        }

    def decompress(self, cue: str, **kwargs) -> str:
        """
        Main method to retrieve and reintegrate knowledge using a cue.
        Corresponds to User Story 5.
        """
        retrieved_items = self._retrieve_and_reintegrate(cue, **kwargs)

        if not retrieved_items:
            return f"No relevant information found for cue: '{cue}'"

        response_parts = [
            f"Retrieved information for cue: '{cue}' (Top {min(3, len(retrieved_items))} shown):"
        ]
        for i, item in enumerate(retrieved_items[:3]):  # Show top 3
            display_content = item["retrieved_content"]
            response_parts.append(
                f"  {i + 1}. Content: '{display_content}' (Confidence: {item['confidence']:.2f}, ID: {item['id']}, Status: {item['status']}, Consol: {item['consolidation_level']:.2f}, Links: {item['linked_traces_count']})"
            )
            if (
                item["original_text"]
                and item["original_text"] != item["retrieved_content"]
            ):  # Show original if different and exists
                orig_preview = item["original_text"]
                if len(orig_preview) > 100:
                    orig_preview = orig_preview[:100] + "..."
                response_parts.append(f"     Original: '{orig_preview}'")

        # Add current working memory context to the response for debugging/demonstration
        response_parts.append(
            f"Current Working Memory Context: {self.working_memory_context}"
        )
        return "\n".join(response_parts)

    # --- Helper methods for each User Story ---

    def _semantic_comprehension(self, text: str) -> dict:
        """
        Simulates User Story 1: Semantic Comprehension of Text.
        """
        print(f"Comprehending: {text}")
        comprehended_info = {"original_text": text}

        # 1. Simulate Semantic Parsing (Simplified)
        tokens = text.split()
        comprehended_info["tokens"] = tokens
        main_tokens = tokens[: min(5, len(tokens))]
        gist = " ".join(main_tokens)
        comprehended_info["main_idea"] = gist.rstrip(".,!?")

        # 2. Context Maintenance
        self.working_memory_context.append(comprehended_info["main_idea"])
        if len(self.working_memory_context) > self.max_context_size:
            self.working_memory_context.pop(0)  # Keep only the last N items
        comprehended_info["current_context"] = list(
            self.working_memory_context
        )  # Store a copy

        # 3. Prior Knowledge (Simplified)
        if tokens and tokens[0] in self.prior_knowledge:
            comprehended_info["related_prior_knowledge"] = self.prior_knowledge[
                tokens[0]
            ]

        # 4. Prediction (Conceptual Placeholder)
        # TODO: Implement prediction based on context and prior knowledge.
        # Example: predict_next_word(self.working_memory_context, self.prior_knowledge)
        comprehended_info["predicted_next_element"] = "..."  # Placeholder

        # 5. Comprehension Monitoring (Conceptual Placeholder)
        # TODO: Implement error detection if text contradicts context or knowledge.
        # Example: check_consistency(text, self.working_memory_context)
        comprehended_info["comprehension_status"] = (
            "nominal"  # Placeholder (e.g., "confused", "error")
        )

        return comprehended_info

    def _short_term_retention(self, comprehended_info: dict) -> dict:
        """
        Simulates User Story 2: Short-Term Retention and Working Memory Management.
        - Focuses on chunking and maintaining info for encoding.
        """
        print(f"Retaining in STM: {comprehended_info.get('main_idea')}")

        # 1. Chunking: The 'main_idea' from comprehension is the primary chunk.
        # More advanced chunking could involve NLP to identify key phrases.
        current_chunk = comprehended_info.get("main_idea", "")

        # Keywords could also be extracted here, e.g., from tokens.
        # For simplicity, we'll rely on the main_idea as the core chunk.

        retained_info = {
            "chunk_content": current_chunk,
            "stm_strength": 1.0,  # Simulate initial strength in STM
            "source_comprehension": comprehended_info,  # Keep original comprehension details
            "status": "retained_in_stm",
        }

        # 2. Limited Capacity & Focus:
        # Implicitly handled by focusing on the 'chunk_content'.
        # The working_memory_context in _semantic_comprehension also models limited capacity.

        # 3. Refreshing Mechanisms (Conceptual):
        # Active rehearsal isn't simulated directly, but stm_strength represents its outcome.
        # If stm_strength were to decay, a rehearsal process would aim to boost it.

        # 4. Organize or "chunk" information - Done by focusing on chunk_content.

        # 5. Ensure a smooth handoff:
        # The returned dictionary is structured for the encoding stage.

        return retained_info

    def _encode_to_long_term_memory(self, retained_info: dict) -> dict:
        """
        Simulates User Story 3: Encoding New Information into Long-Term Memory.
        Creates an initial, hippocampus-dependent memory trace.
        """
        print(f"Encoding to LTM: {retained_info.get('chunk_content')}")

        chunk_content = retained_info.get("chunk_content")
        source_comprehension = retained_info.get("source_comprehension", {})

        ltm_trace = {
            "content": chunk_content,
            "status": "encoded_hippocampal",  # Simulates initial fragile trace
            "encoding_strength": 0.5
            * retained_info.get("stm_strength", 1.0),  # Base strength modulated by STM
            "salience": 0.5,  # Placeholder for novelty/importance
            "encoding_context": {
                "original_text": source_comprehension.get("original_text"),
                "tokens": source_comprehension.get("tokens"),
                "comprehension_context": source_comprehension.get("current_context"),
            },
            "initial_associations": [],
            "consolidation_level": 0.0,  # Will be updated in User Story 4
            "id": self.trace_id_counter,
        }
        self.trace_id_counter += 1

        if "related_prior_knowledge" in source_comprehension:
            ltm_trace["initial_associations"].append(
                source_comprehension["related_prior_knowledge"]
            )

        # TODO: Elaborative encoding could be simulated here by creating more associations
        # or increasing strength based on, e.g., number of tokens or context items.

        self.long_term_memory_store.append(ltm_trace)
        # Return a shallow copy so external modifications don't mutate the store
        return ltm_trace.copy()

    def _consolidate_and_integrate(self):
        """
        Simulates User Story 4: Consolidation and Integration of Memory Over Time.
        Modifies traces in self.long_term_memory_store.
        """
        print("\n--- Starting Consolidation Phase ---")
        consolidation_summary = {
            "strengthened": 0,
            "fully_consolidated": 0,
            "new_links": 0,
        }

        # Create a temporary list of content from all traces for link detection
        trace_contents_for_linking = [
            (trace.get("id"), set(trace.get("content", "").split()))
            for trace in self.long_term_memory_store
        ]

        for trace in self.long_term_memory_store:
            if (
                trace.get("status") == "encoded_hippocampal"
                or trace.get("consolidation_level", 0) < 1.0
            ):
                # Strengthen Trace (modulated by salience)
                consolidation_increment = 0.1 + (0.1 * trace.get("salience", 0.5))
                trace["consolidation_level"] = min(
                    1.0, trace.get("consolidation_level", 0) + consolidation_increment
                )
                trace["encoding_strength"] = min(
                    1.0, trace.get("encoding_strength", 0) + consolidation_increment / 2
                )  # Also slightly boost strength
                consolidation_summary["strengthened"] += 1
                print(
                    f"Consolidating trace ID {trace.get('id')}: '{trace.get('content', '')}' -> new level {trace['consolidation_level']:.2f}"
                )

                # Simulate Transfer if consolidation is high enough
                if (
                    trace["consolidation_level"] >= 0.8
                    and trace.get("status") == "encoded_hippocampal"
                ):
                    trace["status"] = "consolidated_cortical"
                    consolidation_summary["fully_consolidated"] += 1
                    print(
                        f"  Trace ID {trace.get('id')} fully consolidated to cortical."
                    )

                    # Integrate with Existing Knowledge (Simplified Linking)
                    if "linked_traces" not in trace:
                        trace["linked_traces"] = []

                    current_trace_id = trace.get("id")
                    current_content_words = set(trace.get("content", "").split())

                    for other_id, other_content_words in trace_contents_for_linking:
                        if current_trace_id == other_id:
                            continue

                        if current_content_words.intersection(
                            other_content_words
                        ):  # Check for common words
                            # Avoid duplicate links
                            if other_id not in trace["linked_traces"]:
                                trace["linked_traces"].append(other_id)
                                consolidation_summary["new_links"] += 1
                                print(
                                    f"    Linked trace ID {current_trace_id} with trace ID {other_id}"
                                )

                            # Also add link to the other trace
                            for other_trace_for_linking in self.long_term_memory_store:
                                if other_trace_for_linking.get("id") == other_id:
                                    if "linked_traces" not in other_trace_for_linking:
                                        other_trace_for_linking["linked_traces"] = []
                                    if (
                                        current_trace_id
                                        not in other_trace_for_linking["linked_traces"]
                                    ):
                                        other_trace_for_linking["linked_traces"].append(
                                            current_trace_id
                                        )
                                        # No need to double count new_links here, already counted for one side
                                    break

        print(f"--- Consolidation Phase Complete ---")
        print(
            f"Summary: Strengthened {consolidation_summary['strengthened']} traces, "
            f"Fully Consolidated {consolidation_summary['fully_consolidated']} traces, "
            f"New Links {consolidation_summary['new_links']}."
        )

    def trigger_consolidation_phase(self, cycles: int = 1):
        """
        Public method to initiate the consolidation process, simulating e.g. sleep.
        """
        for i in range(cycles):
            print(f"\n=== Running Consolidation Cycle {i + 1}/{cycles} ===")
            self._consolidate_and_integrate()
        # Optionally, return a status or summary from the last cycle.

    def _retrieve_and_reintegrate(self, cue: str, **kwargs) -> list[dict]:
        """
        Simulates User Story 5: Retrieval and Reintegration of Knowledge.
        Searches LTM for traces matching the cue and prepares them for use.
        """
        print(f"\n--- Retrieving and Reintegrating based on cue: '{cue}' ---")

        cue_words = set(cue.lower().split())
        retrieved_candidates = []

        for trace in self.long_term_memory_store:
            trace_content_words = set(trace.get("content", "").lower().split())
            original_text_words = set(
                trace.get("encoding_context", {})
                .get("original_text", "")
                .lower()
                .split()
            )

            # Match if any cue word is in trace content or original text
            if not cue_words.isdisjoint(trace_content_words) or (
                original_text_words and not cue_words.isdisjoint(original_text_words)
            ):

                confidence = (
                    trace.get("consolidation_level", 0)
                    + trace.get("encoding_strength", 0)
                ) / 2
                # Boost confidence for consolidated traces
                if trace.get("status") == "consolidated_cortical":
                    confidence = min(1.0, confidence + 0.2)

                candidate = {
                    "retrieved_content": trace.get("content"),
                    "original_text": trace.get("encoding_context", {}).get(
                        "original_text"
                    ),
                    "id": trace.get("id"),
                    "status": trace.get("status"),
                    "consolidation_level": trace.get("consolidation_level"),
                    "encoding_strength": trace.get("encoding_strength"),
                    "linked_traces_count": len(trace.get("linked_traces", [])),
                    "confidence": round(confidence, 2),  # Metacognitive component
                }
                retrieved_candidates.append(candidate)

        # Sort by confidence
        retrieved_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        if retrieved_candidates:
            # Simulate reintegration into working memory (add top result's content)
            top_retrieved_content = retrieved_candidates[0].get("retrieved_content")
            if top_retrieved_content:
                print(f"Reintegrating to WM: {top_retrieved_content}")
                self.working_memory_context.append(
                    top_retrieved_content
                )  # Add gist/content
                if len(self.working_memory_context) > self.max_context_size:
                    self.working_memory_context.pop(0)

            # Reconsolidation placeholder:
            # For simplicity, we are not implementing active reconsolidation here,
            # but this is where a retrieved trace could become malleable and be updated/re-stored.
            # For example, its strength could be boosted upon successful retrieval and use.
            # top_trace_id = retrieved_candidates[0].get("id")
            # for trace in self.long_term_memory_store:
            # if trace.get("id") == top_trace_id:
            # trace["encoding_strength"] = min(1.0, trace.get("encoding_strength",0) + 0.05) # Minor boost
            # break

        print(f"Found {len(retrieved_candidates)} candidates for cue '{cue}'.")
        return retrieved_candidates

    # TODO: Add other necessary methods and attributes as per the design.


register_compression_engine(NeocortexTransfer.id, NeocortexTransfer, source="contrib")


if __name__ == "__main__":
    # Example Usage (for testing during development)
    engine = NeocortexTransfer()
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_text_2 = "A lazy dog sleeps under the tree."  # Another text for linking

    # Simulate full processing
    print("--- Processing first text ---")
    compressed_representation_1 = engine.compress(sample_text)
    print(f"Compressed output 1: {compressed_representation_1}")

    print("\n--- Processing second text ---")
    compressed_representation_2 = engine.compress(sample_text_2)
    print(f"Compressed output 2: {compressed_representation_2}")

    print(f"\nInitial LTM store: {engine.long_term_memory_store}")

    # Simulate consolidation (can be called separately, e.g., during "sleep")
    engine.trigger_consolidation_phase(cycles=3)  # Simulate 3 consolidation cycles

    print(f"\nLTM store after consolidation:")
    for trace in engine.long_term_memory_store:
        print(trace)

    # Simulate retrieval
    print("\n--- Simulating Retrieval ---")
    print(engine.decompress("fox"))
    print(engine.decompress("lazy dog"))
    print(engine.decompress("moon mission"))  # Should find nothing or low confidence
    print(engine.decompress("unknown concept"))
