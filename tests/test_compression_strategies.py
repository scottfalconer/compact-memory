from compact_memory.compression import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
    NoCompression,
    PipelineCompressionStrategy,
    PipelineStrategyConfig,
    StrategyConfig,
    register_compression_strategy,
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


def test_pipeline_strategy_executes_in_order():
    strat = PipelineCompressionStrategy([DummyStrategy(), DummyStrategy()])
    compressed, trace = strat.compress(
        ["alpha", "bravo", "charlie"], llm_token_budget=5
    )
    assert compressed.text == "alpha bravo"[:5]
    assert len(trace.steps) == 2
    assert trace.output_summary["output_length"] == len(compressed.text)
    for step in trace.steps:
        assert step["strategy"] == DummyStrategy.id
        inner_trace = step["trace"]
        assert isinstance(inner_trace, CompressionTrace)
        assert inner_trace.strategy_name == DummyStrategy.id
        assert inner_trace.output_summary["final_length"] == len(
            inner_trace.final_compressed_object_preview
        )


def test_pipeline_strategy_config_instantiates_from_registry():
    register_compression_strategy(DummyStrategy.id, DummyStrategy)
    cfg = PipelineStrategyConfig(
        strategies=[
            StrategyConfig(strategy_name=DummyStrategy.id),
            StrategyConfig(strategy_name=DummyStrategy.id),
        ]
    )
    strat = cfg.create()
    compressed, trace = strat.compress("alpha bravo charlie", llm_token_budget=5)
    assert compressed.text == "alpha"
    assert len(trace.steps) == 2
