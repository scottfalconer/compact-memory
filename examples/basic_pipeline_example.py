from __future__ import annotations

from compact_memory.compression.strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.registry import register_compression_strategy

# --- Define or Import a Sample Strategy ---
# For this example, we'll define a simple strategy.
# In a real scenario, you might import a strategy from the Compact Memory library
# or a custom strategy package.

class SamplePipelineStrategy(CompressionStrategy):
    """A sample strategy for demonstration in a pipeline."""

    id = "sample_pipeline_strategy"

    def compress(
        self, text_or_chunks: str | list[str], llm_token_budget: int, **kwargs
    ) -> tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses by taking the first `llm_token_budget` characters.
        A real strategy would use a proper tokenizer and more sophisticated logic.
        """
        if isinstance(text_or_chunks, list):
            text_content = " ".join(text_or_chunks)
        else:
            text_content = str(text_or_chunks)

        # Simple truncation based on character count as a proxy for token budget
        # A real implementation should use a tokenizer.
        compressed_text = text_content[:llm_token_budget]

        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"original_length": len(text_content)},
            output_summary={"compressed_length": len(compressed_text)},
            steps=[{"type": "simple_truncation", "details": f"Truncated to {llm_token_budget} chars"}],
        )
        return CompressedMemory(text=compressed_text), trace

# Register the strategy (optional, depending on how you load strategies)
# If this strategy were in its own package, registration would typically happen
# when plugins are loaded.
register_compression_strategy(SamplePipelineStrategy.id, SamplePipelineStrategy)

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
    response = f"LLM response to: '{prompt}' (with context length: {len(compressed_context)})"
    print(f"LLM Response: {response!r}")
    print(f"--- End Mock LLM Call ---")
    return response

# --- Main Example Logic ---
def main():
    print("Starting basic pipeline example...")

    # 1. Initialize the compression strategy
    # We can get the strategy by its ID after it's registered or loaded.
    try:
        strategy_instance = SamplePipelineStrategy()
        # Or, if using the registry:
        # from compact_memory.compression import get_compression_strategy
        # strategy_instance = get_compression_strategy("sample_pipeline_strategy")()
    except Exception as e:
        print(f"Error initializing strategy: {e}")
        print("Please ensure the strategy is defined and registered if needed.")
        return

    # 2. Prepare some text to compress
    original_text = (
        "This is a long piece of text that needs to be compressed "
        "to fit into a limited context window for an LLM. "
        "Compression strategies help reduce the size of the text while "
        "trying to preserve the most important information. "
        "This example demonstrates how to use a strategy programmatically."
    )
    print(f"\nOriginal Text: {original_text!r}")

    # 3. Define a budget for the compressed text (e.g., character count for this mock)
    # In a real scenario, this would be a token budget.
    compression_budget = 100  # Target 100 characters

    # 4. Compress the text using the chosen strategy
    print(f"\nCompressing text with strategy '{strategy_instance.id}' and budget {compression_budget}...")
    try:
        compressed_memory, compression_trace = strategy_instance.compress(
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
