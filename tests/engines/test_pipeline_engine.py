import pytest
from dataclasses import asdict

from compact_memory.engines.pipeline_engine import (
    PipelineEngine,
    PipelineConfig,
    PipelineStepConfig,
)
from compact_memory.engine_config import EngineConfig # Keep this for BaseEngineConfig if needed elsewhere, or remove if not
from typing import Optional
from compact_memory.engines.base import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.engines.no_compression_engine import NoCompressionEngine
from compact_memory.engines.first_last_engine import FirstLastEngine
from unittest import mock

# --- Fixtures ---

@pytest.fixture
def no_op_engine_config() -> PipelineStepConfig: # Changed type hint
    return PipelineStepConfig(engine_name=NoCompressionEngine.id, engine_params={}) # Return PipelineStepConfig

@pytest.fixture
def first_last_engine_config() -> PipelineStepConfig: # Changed type hint
    return PipelineStepConfig(engine_name=FirstLastEngine.id, engine_params={}) # Return PipelineStepConfig

@pytest.fixture
def no_op_engine_instance() -> NoCompressionEngine:
    return NoCompressionEngine()

@pytest.fixture
def first_last_engine_instance() -> FirstLastEngine:
    # Mock the tokenizer for FirstLastEngine for deterministic behavior in tests
    # This avoids dependency on tiktoken being installed or specific tokenization results.
    fle = FirstLastEngine()
    # Patching _DEFAULT_TOKENIZER within first_last_engine module directly
    with mock.patch("compact_memory.engines.first_last_engine._DEFAULT_TOKENIZER", None):
        # Further, ensure tokenize_text (used by FirstLastEngine) uses a simple split,
        # and decode is a simple join for testing purposes.
        # This requires careful patching of where FirstLastEngine gets its tokenizer for decode.
        # The engine's compress method takes a 'tokenizer' kwarg which is used for decoding.
        # The internal tokenization uses compact_memory.token_utils.tokenize_text.
        pass # Mocking will be handled inside tests where specific tokenizer behavior is critical
    return fle


# --- Test Cases ---

def test_pipeline_engine_empty_pipeline():
    text = "Test with empty pipeline."
    budget = 50

    # Test 1: Empty pipeline, no previous_compression_result
    empty_pipeline_config = PipelineConfig(engines=[])
    empty_engine = PipelineEngine(pipeline_definition=empty_pipeline_config)

    result_empty_no_prev = empty_engine.compress(text, budget)

    assert isinstance(result_empty_no_prev, CompressedMemory)
    assert result_empty_no_prev.text == text # Should return original text
    assert result_empty_no_prev.engine_id == PipelineEngine.id
    assert result_empty_no_prev.engine_config is not None
    # Convert PipelineConfig to dict for comparison if engine_config stores it that way
    expected_config_dict = {"budget": budget, "engines": []} # Based on current implementation
    assert result_empty_no_prev.trace.strategy_params.get("budget") == budget # Check some key aspect

    assert isinstance(result_empty_no_prev.trace, CompressionTrace)
    assert result_empty_no_prev.trace.engine_name == PipelineEngine.id
    assert result_empty_no_prev.trace.input_summary == {"original_length": len(text)}
    assert result_empty_no_prev.trace.output_summary == {"compressed_length": len(text)}
    assert len(result_empty_no_prev.trace.steps) == 0
    assert result_empty_no_prev.metadata == {"notes": "Pipeline resulted in no output or was effectively empty."}

    # Test 2: Empty pipeline, with previous_compression_result
    prev_text = "previous text"
    prev_cm = CompressedMemory(text=prev_text, engine_id="prev", trace=CompressionTrace(engine_name="prev", strategy_params={}, input_summary={}, output_summary={}))
    result_empty_with_prev = empty_engine.compress(text, budget, previous_compression_result=prev_cm)

    assert isinstance(result_empty_with_prev, CompressedMemory)
    assert result_empty_with_prev.text == prev_text # Should return previous compressed text
    assert result_empty_with_prev.engine_id == PipelineEngine.id
    # The trace should reflect the pipeline's operation (or lack thereof) on the *original* text for this call
    assert result_empty_with_prev.trace.input_summary == {"original_length": len(text)}
    assert result_empty_with_prev.trace.output_summary == {"compressed_length": len(prev_text)} # Output is from prev_cm
    assert len(result_empty_with_prev.trace.steps) == 0
    # Metadata should be from the prev_cm if the pipeline is empty and prev_cm is passed.
    # Current implementation passes metadata from the *last actual engine run*.
    # If pipeline is empty, current_compressed_memory becomes previous_compression_result.
    assert result_empty_with_prev.metadata == prev_cm.metadata


def test_pipeline_engine_single_engine(no_op_engine_config: EngineConfig, no_op_engine_instance: NoCompressionEngine):
    text = "Test with single NoOp engine."
    budget = 20 # NoOpEngine will truncate if text is longer and budget is smaller

    pipeline_config = PipelineConfig(engines=[no_op_engine_config])
    engine = PipelineEngine(pipeline_definition=pipeline_config)

    result = engine.compress(text, budget)

    assert isinstance(result, CompressedMemory)
    # NoOpEngine with budget might truncate. For this test, assume budget > len(text) or test truncation.
    # Let's use a budget that ensures no truncation by NoOp for simplicity here if not testing truncation.
    budget_no_truncate = len(text) + 10
    result_no_trunc = engine.compress(text, budget_no_truncate)

    assert result_no_trunc.text == text
    assert result_no_trunc.engine_id == PipelineEngine.id

    pipeline_config_dict = asdict(pipeline_config)
    # The engine_config in CompressedMemory for pipeline also includes budget and other kwargs passed to compress.
    # So, it's not just asdict(pipeline_config).
    # Let's check a key part.
    #The engine_config on CompressedMemory is the BaseEngineConfig of the PipelineEngine itself.
    #The pipeline structure is part of the trace's strategy_params.
    assert result_no_trunc.trace.strategy_params.get("engines")[0]["engine_name"] == no_op_engine_config.engine_name

    assert isinstance(result_no_trunc.trace, CompressionTrace)
    assert result_no_trunc.trace.engine_name == PipelineEngine.id
    assert result_no_trunc.trace.input_summary == {"original_length": len(text)}
    assert result_no_trunc.trace.output_summary == {"compressed_length": len(text)}
    assert len(result_no_trunc.trace.steps) == 1

    sub_trace_dict = result_no_trunc.trace.steps[0]
    assert sub_trace_dict["engine_name"] == NoCompressionEngine.id
    assert sub_trace_dict["strategy_params"] == {"budget": budget_no_truncate} # As set by NoOpEngine's trace

def test_pipeline_engine_multiple_engines(no_op_engine_config: PipelineStepConfig, first_last_engine_config: PipelineStepConfig):
    text = "one two three four five six seven eight nine ten" # 10 words
    budget_fle = 4 # For FirstLastEngine: keep first 2, last 2 words. -> "one two nine ten"

    # Pipeline: NoOpEngine -> FirstLastEngine
    # NoOpEngine with large budget won't change the text.
    # FirstLastEngine will then process the original text.
    pipeline_config = PipelineConfig(engines=[no_op_engine_config, first_last_engine_config])
    engine = PipelineEngine(pipeline_definition=pipeline_config)

    # Mock tokenizer for FirstLastEngine part of the pipeline
    # This is complex because the instance of FirstLastEngine is created inside PipelineEngine.
    # We'd have to mock where get_compression_engine / EngineConfig.create instantiates it.
    # Simpler: Rely on FirstLastEngine's fallback to split() if tiktoken is not available,
    # or ensure tests for FirstLastEngine itself cover tokenizer variations.
    # For this pipeline test, focus on data flow.
    # We'll assume FirstLastEngine is configured/mocked to behave predictably (e.g. uses split())

    # To make FirstLastEngine predictable without complex mocking here,
    # we can instantiate it manually with a mocked/simple tokenizer and pass list of instances
    mocked_fle = FirstLastEngine()

    # We need to ensure the FirstLastEngine instance within the pipeline uses a mock tokenizer for decode.
    # This is tricky. Let's assume it falls back to string split and join for this test for simplicity of setup.
    # If _DEFAULT_TOKENIZER is None in first_last_engine.py, it uses split() for tokenization part.
    # The decode part is `tokenizer.decode`. If we pass `tokenizer=str.split` to FLE.compress, it fails.
    # The `tokenizer` param in FLE.compress is for the *decode* step.
    # The tokenization part uses `compact_memory.token_utils.tokenize_text`

    # For pipeline, it's easier to test if sub-engines are simple or have globally patched tokenizers.
    # Let's assume the simple split/join for FLE for this test.

    # Create instances for the pipeline
    no_op_inst = NoCompressionEngine()

    # For FirstLastEngine, we need its tokenizer to be mocked for predictable output
    # Patching the _DEFAULT_TOKENIZER in the module FirstLastEngine uses
    with mock.patch("compact_memory.engines.first_last_engine._DEFAULT_TOKENIZER", None):

        # Create FirstLastEngine instance *after* patching, so it picks up the mocked default
        fle_inst = FirstLastEngine()

        # Now create pipeline with instances
        engine_instances = PipelineEngine(pipeline_definition=[no_op_inst, fle_inst])

        # The 'tokenizer' argument to engine.compress is for the sub-engines if they accept it.
        # FirstLastEngine's compress accepts 'tokenizer' for its decode step.
        # NoOpEngine's compress also accepts 'tokenizer'.
        # PipelineEngine passes kwargs including 'tokenizer' to its sub-engines.

        # Define a mock decode function to be passed as 'tokenizer' argument
        def mock_decode_func(tokens): return " ".join(tokens)

        result = engine_instances.compress(text, budget_fle, tokenizer=mock_decode_func)

        assert isinstance(result, CompressedMemory)
        expected_text_after_fle = "one two nine ten"
        assert result.text == expected_text_after_fle
        assert result.engine_id == PipelineEngine.id

        assert isinstance(result.trace, CompressionTrace)
        assert result.trace.engine_name == PipelineEngine.id
        assert len(result.trace.steps) == 2

        # Check NoOpEngine's trace (first step)
        noop_trace_dict = result.trace.steps[0]
        assert noop_trace_dict["engine_name"] == NoCompressionEngine.id
        # NoOpEngine's output (which is input to FLE) should be the original text
        # This is hard to check directly from trace dicts unless output is part of trace.
        # We can check strategy_params.
        assert noop_trace_dict["strategy_params"]["budget"] == budget_fle # budget is passed along

        # Check FirstLastEngine's trace (second step)
        fle_trace_dict = result.trace.steps[1]
        assert fle_trace_dict["engine_name"] == FirstLastEngine.id
        assert fle_trace_dict["strategy_params"]["budget"] == budget_fle
        assert fle_trace_dict["input_summary"]["input_tokens"] == 10 # "one two three four five six seven eight nine ten"
        assert fle_trace_dict["output_summary"]["final_tokens"] == 4  # "one two nine ten"

def test_pipeline_engine_with_initial_previous_compression(no_op_engine_config: EngineConfig):
    initial_text = "Initial compressed text by a previous process."
    initial_trace = CompressionTrace(engine_name="initial_compressor", strategy_params={}, input_summary={}, output_summary={})
    initial_cm = CompressedMemory(text=initial_text, engine_id="initial", trace=initial_trace)

    pipeline_config = PipelineConfig(engines=[no_op_engine_config]) # Simple pipeline with one NoOp
    engine = PipelineEngine(pipeline_definition=pipeline_config)

    # This text is the "new" text the pipeline is asked to compress, but it should use initial_cm.text for the first engine
    new_text_for_pipeline = "This is new text, but pipeline should use previous_compression_result's text."
    budget = len(initial_text) + 10 # Ensure NoOp doesn't truncate initial_text

    result = engine.compress(new_text_for_pipeline, budget, previous_compression_result=initial_cm)

    assert isinstance(result, CompressedMemory)
    # The NoOp engine will receive initial_cm.text. Since budget is ample, it won't change it.
    assert result.text == initial_text
    assert result.engine_id == PipelineEngine.id

    assert isinstance(result.trace, CompressionTrace)
    assert result.trace.engine_name == PipelineEngine.id
    # The pipeline's input_summary should reflect the new_text_for_pipeline, not initial_cm's length
    assert result.trace.input_summary == {"original_length": len(new_text_for_pipeline)}
    # The pipeline's output_summary should reflect the final text length
    assert result.trace.output_summary == {"compressed_length": len(initial_text)}
    assert len(result.trace.steps) == 1

    # The sub-trace for NoOpEngine
    noop_sub_trace = result.trace.steps[0]
    assert noop_sub_trace["engine_name"] == NoCompressionEngine.id
    # Input to NoOpEngine was initial_text
    assert noop_sub_trace["input_summary"]["input_length"] == len(initial_text)
    assert noop_sub_trace["output_summary"]["output_length"] == len(initial_text)

def test_pipeline_engine_passes_previous_result_between_engines():
    # Create mock engines that record the previous_compression_result they received
    class RecorderEngine(BaseCompressionEngine):
        id = "recorder"
        def __init__(self, name="default", **kwargs):
            super().__init__(**kwargs)
            self.received_previous_cm = None
            self.name = name # To distinguish instances

        def compress(self, text: str, budget: int, previous_compression_result: Optional[CompressedMemory] = None, **kwargs) -> CompressedMemory:
            self.received_previous_cm = previous_compression_result
            # Return a new CM, possibly modifying text or just passing it along
            new_text = f"{self.name}_processed_{text}"
            trace = CompressionTrace(engine_name=self.id, strategy_params={"budget":budget}, input_summary={"original_length":len(text)}, output_summary={"compressed_length":len(new_text)})
            return CompressedMemory(text=new_text, engine_id=self.id, engine_config=self.config, trace=trace, metadata={"processed_by": self.name})

    engine1 = RecorderEngine(name="eng1")
    engine2 = RecorderEngine(name="eng2")

    pipeline = PipelineEngine(pipeline_definition=[engine1, engine2])

    original_text = "start"
    budget = 100

    # Test 1: No initial previous_compression_result
    final_result = pipeline.compress(original_text, budget)

    assert engine1.received_previous_cm is None, "Engine 1 should not have received a previous_compression_result"

    # Engine 2 should have received the CompressedMemory object from Engine 1
    assert engine2.received_previous_cm is not None
    assert isinstance(engine2.received_previous_cm, CompressedMemory)
    assert engine2.received_previous_cm.text == "eng1_processed_start" # Output of engine1
    assert engine2.received_previous_cm.engine_id == "recorder" # engine1.id
    assert engine2.received_previous_cm.metadata == {"processed_by": "eng1"}

    assert final_result.text == "eng2_processed_eng1_processed_start"

    # Test 2: With an initial previous_compression_result
    initial_cm_text = "initial_cm_content"
    initial_cm = CompressedMemory(text=initial_cm_text, engine_id="initial", trace=CompressionTrace(engine_name="initial", strategy_params={}, input_summary={}, output_summary={}))

    # Reset received_previous_cm for engine instances if they are stateful (they are here)
    engine1.received_previous_cm = None
    engine2.received_previous_cm = None

    final_result_with_initial = pipeline.compress("new_text_ignored_for_first_step", budget, previous_compression_result=initial_cm)

    # Engine 1 should receive the initial_cm
    assert engine1.received_previous_cm is not None
    assert engine1.received_previous_cm.text == initial_cm_text
    assert engine1.received_previous_cm.engine_id == "initial"

    # Engine 2 should receive the result from Engine 1 (which processed initial_cm_text)
    assert engine2.received_previous_cm is not None
    assert engine2.received_previous_cm.text == f"eng1_processed_{initial_cm_text}"
    assert engine2.received_previous_cm.engine_id == "recorder"
    assert engine2.received_previous_cm.metadata == {"processed_by": "eng1"}

    assert final_result_with_initial.text == f"eng2_processed_eng1_processed_{initial_cm_text}"
