from CompressionEngine.core.registry import ( # Updated path
    register_compression_engine, # Updated function name
    _COMPRESSION_REGISTRY,
)
from CompressionEngine.core.engines_abc import CompressionEngine # Updated path and class name
from compact_memory.validation.registry import (
    register_validation_metric,
    get_validation_metric_class,
    _VALIDATION_METRIC_REGISTRY,
)
from compact_memory.validation.metrics_abc import ValidationMetric

from CompressionEngine.core.engines_abc import ( # Updated path
    CompressedMemory,
    CompressionTrace,
)


def test_register_compression_engine(): # Updated function name
    class DummyEngine(CompressionEngine): # Updated class name
        id = "dummy_engine" # Updated id

        def compress(self, text_or_chunks, llm_token_budget, **kwargs):
            if isinstance(text_or_chunks, list):
                text = " ".join(text_or_chunks)
            else:
                text = text_or_chunks
            text = text[:llm_token_budget] if llm_token_budget else text
            compressed = CompressedMemory(text=text)
            trace = CompressionTrace(
                engine_name=self.id, # Updated parameter name
                engine_params={"llm_token_budget": llm_token_budget}, # Updated parameter name
                input_summary={
                    "input_length": len(
                        text_or_chunks
                        if isinstance(text_or_chunks, str)
                        else " ".join(text_or_chunks)
                    )
                },
            )
            return compressed, trace

    register_compression_engine(DummyEngine.id, DummyEngine) # Updated function and class names
    assert _COMPRESSION_REGISTRY["dummy_engine"] is DummyEngine # Updated key and class name
    compressed, trace = DummyEngine().compress("alpha bravo", llm_token_budget=5) # Updated class name
    assert isinstance(compressed, CompressedMemory)
    assert isinstance(trace, CompressionTrace)


def test_register_validation_metric():
    class DummyMetric(ValidationMetric):
        id = "metric"

        def evaluate(self, llm_response: str, reference_answer: str, **kw):
            return {"score": 0.0}

    register_validation_metric(DummyMetric.id, DummyMetric)
    assert _VALIDATION_METRIC_REGISTRY["metric"] is DummyMetric
    assert get_validation_metric_class("metric") is DummyMetric
