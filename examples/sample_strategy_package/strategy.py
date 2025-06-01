from gist_memory.compression.strategies_abc import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)


class SampleStrategy(CompressionStrategy):
    """Trivial example strategy that returns the input unchanged."""

    id = "sample"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        text = text_or_chunks if isinstance(text_or_chunks, str) else " ".join(text_or_chunks)
        return CompressedMemory(text=text), CompressionTrace()
