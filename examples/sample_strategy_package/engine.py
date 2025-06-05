from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)


class SampleEngine(BaseCompressionEngine):
    """Trivial example engine that returns the input unchanged."""

    # Unique string identifier for this engine. This ID is used by the Compact Memory framework to find and load the engine.
    id = "sample"

    def compress(
        self, text_or_chunks: str | list[str], llm_token_budget: int, **kwargs: object
    ) -> tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text or chunks to meet the token budget.

        Args:
            text_or_chunks: The input text (as a single string) or a list of text chunks to be compressed.
            llm_token_budget: The target maximum number of tokens (or a proxy like characters,
                              depending on the engine's implementation) that the compressed
                              output should ideally have. The engine should aim to stay within this budget.
            **kwargs: Additional keyword arguments that might be passed by the framework or user.
                      A common one is `tokenizer`, which might be a tokenizer function/object
                      that can be used to count tokens or process text.
                      Example: `tokenizer = kwargs.get("tokenizer")`

        Returns:
            A tuple containing:
                - CompressedMemory: An object with the `text` attribute holding the compressed
                                  string result. It can also optionally include `metadata`.
                - CompressionTrace: An object that logs details about the compression process
                                  (e.g., strategy name, parameters, input/output token counts,
                                  steps taken). This is useful for debugging and analysis.
                                  Even for simple strategies, providing a basic trace is good practice.
        """
        # If input is a list of chunks, join them. A more sophisticated engine might process chunks individually.
        text = (
            text_or_chunks
            if isinstance(text_or_chunks, str)
            else " ".join(text_or_chunks)
        )

        # --- Your actual compression logic would go here ---
        # This sample engine is trivial and does no real compression.
        # It just returns the original text, potentially truncated if it were to respect the budget.
        # A real engine would:
        # 1. Use the tokenizer (if provided via kwargs) to count tokens.
        # 2. Implement an algorithm to select, summarize, or transform the text to reduce its token count
        #    to be at or below `llm_token_budget`.
        # 3. Populate the CompressionTrace with meaningful information.

        compressed_output_text = text  # Placeholder

        # Create a CompressedMemory object containing the result.
        memory = CompressedMemory(text=compressed_output_text)

        # Create a CompressionTrace object.
        # For a real engine, you'd populate this with more details about the process.
        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={
                "llm_token_budget": llm_token_budget
            },  # Log key parameters
            input_summary={
                "original_length": len(text),
                "original_tokens": kwargs.get("tokenizer", lambda x: len(x.split()))(
                    text
                ).get("input_ids", []),
            },  # Example token count
            output_summary={
                "compressed_length": len(compressed_output_text),
                "compressed_tokens": kwargs.get("tokenizer", lambda x: len(x.split()))(
                    compressed_output_text
                ).get("input_ids", []),
            },
            steps=[
                {"type": "sample_noop", "details": "Sample engine returned text as is."}
            ],
        )
        return memory, trace
