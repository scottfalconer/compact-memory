from gist_memory.compression import CompressionStrategy, CompressedMemory


class DummyStrategy(CompressionStrategy):
    id = "dummy"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = text_or_chunks
        return CompressedMemory(text=text[:llm_token_budget])


def test_dummy_strategy():
    strat = DummyStrategy()
    result = strat.compress(["alpha", "bravo"], llm_token_budget=10)
    assert isinstance(result, CompressedMemory)
    assert result.text == "alpha bravo"[:10]

