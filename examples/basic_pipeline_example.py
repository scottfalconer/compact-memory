from __future__ import annotations

from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.engine_registry import register_compression_engine

# --- Define or Import a Sample Engine ---
# For this example, we'll define a simple engine.
# In a real scenario, you might import an engine from the Compact Memory library
# or a custom engine package.


class SamplePipelineEngine(BaseCompressionEngine):
    """A sample engine for demonstration in a pipeline."""

    id = "sample_pipeline_engine"

    def compress(
        self, text_or_chunks: str | list[str], llm_token_budget: int, **kwargs
    ) -> tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses by taking the first `llm_token_budget` characters.
        A real engine would use a proper tokenizer and more sophisticated logic.
        """
        if isinstance(text_or_chunks, list):
            text_content = " ".join(text_or_chunks)
        else:
            text_content = str(text_or_chunks)

        # Simple truncation based on character count as a proxy for token budget
        # A real implementation should use a tokenizer.
        compressed_text = text_content[:llm_token_budget]

        trace = CompressionTrace(
            engine_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"original_length": len(text_content)},
            output_summary={"compressed_length": len(compressed_text)},
            steps=[
                {
                    "type": "simple_truncation",
                    "details": f"Truncated to {llm_token_budget} chars",
                }
            ],
        )
        return CompressedMemory(text=compressed_text), trace


# Register the engine (optional, depending on how you load engines)
# If this engine were in its own package, registration would typically happen
# when plugins are loaded.
register_compression_engine(SamplePipelineEngine.id, SamplePipelineEngine)


# --- Mock LLM Function ---
def mock_llm_call(prompt: str, compressed_context: str = "") -> str:
    """
    Simulates calling a Language Model.
    In a real application, this would interact with an actual LLM API.
    """
    print(f"\n--- Mock LLM Call ---")
    if compressed_context:
        print(f"Compressed Context: {compressed_context!r}")
    print(f"Prompt: {prompt!r}")
    response = (
        f"LLM response to: '{prompt}' (with context length: {len(compressed_context)})"
    )
    print(f"LLM Response: {response!r}")
    print(f"--- End Mock LLM Call ---")
    return response


# --- Main Example Logic ---
def main():
    print("Starting basic pipeline example...")

    # 1. Initialize the compression engine
    # We can get the engine by its ID after it's registered or loaded.
    try:
        engine_instance = SamplePipelineEngine()
        # Or, if using the registry:
        # from compact_memory.engine_registry import get_compression_engine
        # engine_instance = get_compression_engine("sample_pipeline_engine")()
    except Exception as e:
        print(f"Error initializing engine: {e}")
        print("Please ensure the engine is defined and registered if needed.")
        return

    # 2. Prepare some text to compress
    original_text = (
        "This is a long piece of text that needs to be compressed "
        "to fit into a limited context window for an LLM. "
        "Compression engines help reduce the size of the text while "
        "trying to preserve the most important information. "
        "This example demonstrates how to use an engine programmatically."
    )
    print(f"\nOriginal Text: {original_text!r}")

    # 3. Define a budget for the compressed text (e.g., character count for this mock)
    # In a real scenario, this would be a token budget.
    compression_budget = 100  # Target 100 characters

    # 4. Compress the text using the chosen engine
    print(
        f"\nCompressing text with engine '{engine_instance.id}' and budget {compression_budget}..."
    )
    try:
        compressed_memory, compression_trace = engine_instance.compress(
            original_text, llm_token_budget=compression_budget
        )
        compressed_text = compressed_memory.text
        print(f"Compressed Text: {compressed_text!r}")
        print(f"Compression Trace: {compression_trace}")
    except Exception as e:
        print(f"Error during compression: {e}")
        return

    # 5. Use the compressed text in a (mock) LLM call
    # This demonstrates feeding the compressed output as context to an LLM.
    user_query = "What is the main purpose of this text?"

    # Scenario 1: LLM call with compressed context
    print("\nScenario 1: LLM call with compressed_text as context.")
    mock_llm_call(prompt=user_query, compressed_context=compressed_text)

    # Scenario 2: LLM call where compressed text might be part of a larger prompt
    print("\nScenario 2: LLM call where compressed_text is integrated into the prompt.")
    combined_prompt = (
        f"Based on the following summary: '{compressed_text}', "
        f"answer this question: {user_query}"
    )
    mock_llm_call(prompt=combined_prompt)

    print("\nBasic pipeline example finished.")


if __name__ == "__main__":
    main()
