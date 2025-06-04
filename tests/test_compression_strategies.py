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


def test_dummy_strategy_trace_details():
    """Ensure DummyStrategy records detailed trace metadata."""
    strat = DummyStrategy()
    text = "alpha bravo"
    compressed, trace = strat.compress(text, llm_token_budget=5)

    assert trace.strategy_name == "dummy"
    assert trace.strategy_params["llm_token_budget"] == 5
    assert trace.input_summary["input_length"] == len(text)
    assert trace.steps == [{"type": "truncate"}]
    assert trace.output_summary["final_length"] == len(compressed.text)
    assert trace.final_compressed_object_preview == compressed.text


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


def test_no_compression_trace_details():
    """Validate trace information for NoCompression."""
    tokenizer = SimpleTokenizer()
    text = "alpha bravo charlie"
    compressed, trace = NoCompression().compress(
        text,
        llm_token_budget=2,
        tokenizer=tokenizer,
    )

    assert trace.strategy_name == "none"
    assert trace.input_summary["input_length"] == len(compressed.text)
    assert trace.output_summary["output_length"] == len(compressed.text)
    assert trace.steps == []
    assert trace.final_compressed_object_preview is None


def test_pipeline_strategy_executes_in_order():
    strat = PipelineCompressionStrategy([DummyStrategy(), DummyStrategy()])
    compressed, trace = strat.compress(
        ["alpha", "bravo", "charlie"], llm_token_budget=5
    )
    assert compressed.text == "alpha bravo"[:5]
    assert len(trace.steps) == 2


def test_pipeline_trace_contains_step_details():
    """Verify nested step traces are recorded correctly."""
    strat = PipelineCompressionStrategy([DummyStrategy(), DummyStrategy()])
    compressed, trace = strat.compress(
        ["alpha", "bravo", "charlie"], llm_token_budget=5
    )

    assert compressed.text == "alpha"
    assert len(trace.steps) == 2

    first = trace.steps[0]
    second = trace.steps[1]

    assert first["strategy"] == "dummy"
    assert isinstance(first["trace"], CompressionTrace)
    assert first["trace"].output_summary["final_length"] == 5

    assert second["strategy"] == "dummy"
    assert isinstance(second["trace"], CompressionTrace)
    assert second["trace"].input_summary["input_length"] == 5
    assert second["trace"].steps == [{"type": "truncate"}]


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
