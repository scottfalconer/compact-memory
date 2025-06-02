from gist_memory.compression import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
    NoCompression,
    ImportanceCompression,
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


class SimpleTokenizer:
    def tokenize(self, text):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_no_compression_truncates_and_traces():
    tokenizer = SimpleTokenizer()
    strat = NoCompression()
    compressed, trace = strat.compress(
        ["alpha", "bravo", "charlie"],
        llm_token_budget=2,
        tokenizer=tokenizer,
    )
    assert compressed.text == "alpha bravo"
    assert trace.strategy_name == "none"


def test_importance_compression_filters_and_truncates():
    tokenizer = SimpleTokenizer()
    strat = ImportanceCompression()
    text = "Bob: hello\nuh-huh\nAlice: hi"
    compressed, trace = strat.compress(text, llm_token_budget=4, tokenizer=tokenizer)
    assert "uh-huh" not in compressed.text
    assert compressed.text.split()[:4] == compressed.text.split()
    assert trace.strategy_name == "importance"

