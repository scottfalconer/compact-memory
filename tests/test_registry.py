from compact_memory.engines.registry import (
    register_compression_engine,
    _ENGINE_REGISTRY,
)
from compact_memory.engines import (
    BaseCompressionEngine as CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.validation.registry import (
    register_validation_metric,
    get_validation_metric_class,
    _VALIDATION_METRIC_REGISTRY,
)
from compact_memory.validation.metrics_abc import ValidationMetric


def test_register_compression_strategy():
    class DummyStrategy(CompressionStrategy):
        id = "dummy"

        def compress(self, text_or_chunks, llm_token_budget, **kwargs):
            if isinstance(text_or_chunks, list):
                text = " ".join(text_or_chunks)
            else:
                text = text_or_chunks
            text = text[:llm_token_budget] if llm_token_budget else text
            compressed = CompressedMemory(text=text)
            trace = CompressionTrace(
                strategy_name=self.id,
                strategy_params={"llm_token_budget": llm_token_budget},
                input_summary={
                    "input_length": len(
                        text_or_chunks
                        if isinstance(text_or_chunks, str)
                        else " ".join(text_or_chunks)
                    )
                },
            )
            return compressed, trace

    register_compression_engine(DummyStrategy.id, DummyStrategy)
    assert _ENGINE_REGISTRY["dummy"] is DummyStrategy
    compressed, trace = DummyStrategy().compress("alpha bravo", llm_token_budget=5)
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
