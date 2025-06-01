from gist_memory.compression import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)


class DummyStrategy(CompressionStrategy):
    id = "dummy"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = text_or_chunks
        compressed = CompressedMemory(text=text[:llm_token_budget])
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            steps=[{"type": "truncate"}],
            output_summary={"final_length": len(compressed.text)},
            final_compressed_object_preview=compressed.text,
        )
        return compressed, trace


def test_dummy_strategy():
    strat = DummyStrategy()
    compressed, trace = strat.compress(["alpha", "bravo"], llm_token_budget=10)
    assert isinstance(compressed, CompressedMemory)
    assert isinstance(trace, CompressionTrace)
    assert compressed.text == "alpha bravo"[:10]

