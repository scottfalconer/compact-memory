# --- Advanced compress_text Example: Inspecting Trace and References ---

from compact_memory import compress_text
from compact_memory.token_utils import get_tokenizer, Tokenizer # Tokenizer for type hint
from compact_memory.api_models import CompressedMemoryContext, SourceReference # For type hinting if needed
import json # For pretty printing the trace
import os

# It's useful to pick a strategy that generates somewhat interesting trace or references.
# FirstLastStrategy is good for demonstrating source_references clearly.

class DummyTokenizer:
    """A fallback tokenizer if a real one cannot be loaded."""
    def __init__(self, name="dummy"):
        self.name = name
        print(f"Warning: Using DummyTokenizer ({name}). Token counts will be approximate (word count).")

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, text: str) -> list:
        return list(text.split()) # Not real tokens, just for structure

    def decode(self, tokens: list) -> str:
        return " ".join(tokens)


def run_advanced_compression_example():
    """Demonstrates advanced inspection of compress_text output."""

    sample_text = (
        "The first line of an important document which contains introductory material.\n"
        "The second line, containing further details and elaborating on the first.\n"
        "A middle section that might be considered less critical for a quick summary.\n"
        "Another part of the middle, perhaps redundant if space is very limited.\n"
        "The penultimate line, which often carries concluding remarks or key takeaways.\n"
        "The very final line, essential for understanding the absolute end of the document."
    )

    print(f"--- Original Text ---\n{sample_text}\n")

    tokenizer_instance: Tokenizer
    try:
        # Using a real tokenizer is important for strategies that budget by tokens.
        default_tokenizer_name = os.environ.get("CM_TEST_TOKENIZER", "gpt2")
        tokenizer_instance = get_tokenizer(default_tokenizer_name)
        print(f"Using tokenizer: {default_tokenizer_name}\n")
    except Exception as e:
        print(f"Could not load '{os.environ.get('CM_TEST_TOKENIZER', 'gpt2')}' tokenizer: {e}")
        tokenizer_instance = DummyTokenizer()

    strategy_class_id = "FirstLastStrategy"
    # Parameters: get first 2 lines, last 2 lines, custom separator
    strategy_params = {"first_k": 2, "last_k": 2, "separator": " [...SNIP...] "}
    # Budget for FirstLastStrategy often relates to the number of "k" items,
    # but can also be used by its internal budgeting if it truncates lines to fit a token limit.
    budget = 100 # Target token budget (actual effect depends on strategy's budgeting logic)

    try:
        compressed_context = compress_text(
            text=sample_text,
            strategy_class_id=strategy_class_id,
            budget=budget,
            strategy_params=strategy_params,
            tokenizer_instance=tokenizer_instance,
            llm_provider_instance=None # FirstLastStrategy does not need an LLM
        )

        print(f"--- Compressed Output (Strategy ID Used: {compressed_context.strategy_id_used}) ---")
        print(f"Compressed Text:\n{compressed_context.compressed_text}\n")

        print("--- Source References ---")
        if compressed_context.source_references:
            for i, ref in enumerate(compressed_context.source_references):
                print(f"  Reference {i+1}:")
                print(f"    Snippet: '{ref.text_snippet.strip()}'")
                if ref.document_id: print(f"    Doc ID: {ref.document_id}") # Will be 'original_input_text'
                if ref.chunk_id: print(f"    Chunk ID: {ref.chunk_id}") # Strategy might populate this
                if ref.score is not None: print(f"    Score: {ref.score}")
                if ref.metadata: print(f"    Metadata: {ref.metadata}") # Strategy might add original line numbers etc.
        else:
            print("  No detailed source references provided by the strategy for this input.")
        print("") # Newline for spacing

        print("--- Budget Information ---")
        if compressed_context.budget_info:
            print(f"  Requested Budget: {compressed_context.budget_info.get('requested_budget')}")
            # These might be in full_trace for more detail depending on strategy
            print(f"  Final Tokens (Reported by API Model): {compressed_context.budget_info.get('final_tokens')}")
        else:
            print("  No detailed budget information available in budget_info.")
        print("")

        print("--- Processing Time ---")
        print(f"  Reported processing time: {compressed_context.processing_time_ms:.2f} ms\n")

        print("--- Full Compression Trace (JSON) ---")
        if compressed_context.full_trace:
            # Pretty print the JSON trace
            trace_dict = compressed_context.full_trace
            print(json.dumps(trace_dict, indent=2))
            # Example: Access specific trace details if known
            if 'original_tokens' in trace_dict:
                 print(f"  Trace - Original Tokens: {trace_dict['original_tokens']}")
            if 'compressed_tokens' in trace_dict:
                 print(f"  Trace - Compressed Tokens: {trace_dict['compressed_tokens']}")
            if 'steps' in trace_dict and isinstance(trace_dict['steps'], list):
                print(f"  Trace - Number of Steps: {len(trace_dict['steps'])}")
        else:
            print("  No detailed trace information provided.")

    except Exception as e:
        print(f"An error occurred during advanced compression: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_advanced_compression_example()
```
