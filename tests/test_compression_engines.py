from CompressionEngine.core import ( # Updated path
    CompressionEngine, # Updated class name
    CompressedMemory,
    CompressionTrace,
    NoCompressionEngine, # Updated class name
    PipelineCompressionEngine, # Updated class name
    PipelineEngineConfig, # Updated class name
    EngineConfig, # Updated class name
    register_compression_engine, # Updated function name
)


class DummyEngine(CompressionEngine): # Updated class name
    id = "dummy_engine" # Updated id

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = text_or_chunks
        compressed = CompressedMemory(text=text[:llm_token_budget])
        trace = CompressionTrace(
            engine_name=self.id, # Updated parameter name
            engine_params={"llm_token_budget": llm_token_budget}, # Updated parameter name
            input_summary={"input_length": len(text)},
            steps=[{"type": "truncate"}],
            output_summary={"final_length": len(compressed.text)},
            final_compressed_object_preview=compressed.text,
        )
        return compressed, trace


def test_dummy_engine(): # Updated function name
    engine = DummyEngine() # Updated variable and class name
    compressed, trace = engine.compress(["alpha", "bravo"], llm_token_budget=10) # Updated variable name
    assert isinstance(compressed, CompressedMemory)
    assert isinstance(trace, CompressionTrace)
    assert compressed.text == "alpha bravo"[:10]


def test_dummy_engine_trace_details(): # Updated function name
    """Ensure DummyEngine records detailed trace metadata.""" # Updated docstring
    engine = DummyEngine() # Updated variable and class name
    text = "alpha bravo"
    compressed, trace = engine.compress(text, llm_token_budget=5) # Updated variable name

    assert trace.engine_name == "dummy_engine" # Updated parameter name and value
    assert trace.engine_params["llm_token_budget"] == 5 # Updated parameter name
    assert trace.input_summary["input_length"] == len(text)
    assert trace.steps == [{"type": "truncate"}]
    assert trace.output_summary["final_length"] == len(compressed.text)
    assert trace.final_compressed_object_preview == compressed.text


class SimpleTokenizer:
    def tokenize(self, text):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_no_compression_engine_truncates_and_traces(): # Updated function name
    tokenizer = SimpleTokenizer()
    engine = NoCompressionEngine() # Updated variable and class name
    compressed, trace = engine.compress( # Updated variable name
        ["alpha", "bravo", "charlie"],
        llm_token_budget=2,
        tokenizer=tokenizer,
    )
    assert compressed.text == "alpha bravo"
    assert trace.engine_name == "no_compression_engine" # Updated parameter name and value


def test_no_compression_engine_trace_details(): # Updated function name
    """Validate trace information for NoCompressionEngine.""" # Updated docstring
    tokenizer = SimpleTokenizer()
    text = "alpha bravo charlie"
    compressed, trace = NoCompressionEngine().compress( # Updated class name
        text,
        llm_token_budget=2,
        tokenizer=tokenizer,
    )

    assert trace.engine_name == "no_compression_engine" # Updated parameter name and value
    assert trace.input_summary["input_length"] == len(compressed.text)
    assert trace.output_summary["output_length"] == len(compressed.text)
    assert trace.steps == []
    assert trace.final_compressed_object_preview is None


def test_pipeline_engine_executes_in_order(): # Updated function name
    engine = PipelineCompressionEngine([DummyEngine(), DummyEngine()]) # Updated variable and class names
    compressed, trace = engine.compress( # Updated variable name
        ["alpha", "bravo", "charlie"], llm_token_budget=5
    )
    assert compressed.text == "alpha bravo"[:5] # This assertion seems to be different from the one below, check if this is intended.
    assert len(trace.steps) == 2
    assert trace.output_summary["output_length"] == len(compressed.text)
    for step in trace.steps:
        assert step["engine"] == DummyEngine.id # Updated key and class name
        inner_trace = step["trace"]
        assert isinstance(inner_trace, CompressionTrace)
        assert inner_trace.engine_name == DummyEngine.id # Updated parameter and class name
        assert inner_trace.output_summary["final_length"] == len(
            inner_trace.final_compressed_object_preview
        )


def test_pipeline_engine_trace_contains_step_details(): # Updated function name
    """Verify nested step traces are recorded correctly."""
    engine = PipelineCompressionEngine([DummyEngine(), DummyEngine()]) # Updated variable and class names
    compressed, trace = engine.compress( # Updated variable name
        ["alpha", "bravo", "charlie"], llm_token_budget=5
    )

    assert compressed.text == "alpha"
    assert len(trace.steps) == 2

    first = trace.steps[0]
    second = trace.steps[1]

    assert first["engine"] == "dummy_engine" # Updated key and value
    assert isinstance(first["trace"], CompressionTrace)
    assert first["trace"].output_summary["final_length"] == 5

    assert second["engine"] == "dummy_engine" # Updated key and value
    assert isinstance(second["trace"], CompressionTrace)
    assert second["trace"].input_summary["input_length"] == 5
    assert second["trace"].steps == [{"type": "truncate"}]


def test_pipeline_engine_config_instantiates_from_registry(): # Updated function name
    register_compression_engine(DummyEngine.id, DummyEngine) # Updated function and class names
    cfg = PipelineEngineConfig( # Updated class name
        engines=[ # Updated field name
            EngineConfig(engine_name=DummyEngine.id), # Updated class name and field name
            EngineConfig(engine_name=DummyEngine.id), # Updated class name and field name
        ]
    )
    engine = cfg.create() # Updated variable name
    compressed, trace = engine.compress("alpha bravo charlie", llm_token_budget=5) # Updated variable name
    assert compressed.text == "alpha"
    assert len(trace.steps) == 2
