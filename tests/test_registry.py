from gist_memory.registry import (
    CompressionStrategy,
    ValidationMetric,
    register_compression_strategy,
    register_validation_metric,
    _COMPRESSION_REGISTRY,
    _VALIDATION_REGISTRY,
)


def test_register_compression_strategy():
    class DummyStrategy(CompressionStrategy):
        id = "dummy"

        def compress(self, text: str) -> str:
            return text

    register_compression_strategy(DummyStrategy.id, DummyStrategy)
    assert _COMPRESSION_REGISTRY["dummy"] is DummyStrategy


def test_register_validation_metric():
    class DummyMetric(ValidationMetric):
        id = "metric"

        def compute(self, reference: str, prediction: str) -> float:
            return 0.0

    register_validation_metric(DummyMetric.id, DummyMetric)
    assert _VALIDATION_REGISTRY["metric"] is DummyMetric
