from compact_memory.engine_registry import (
    register_compression_engine,
    _ENGINE_REGISTRY,
)
from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)


def test_register_compression_engine() -> None:
    class DummyEngine(BaseCompressionEngine):
        id = "dummy_engine"

    register_compression_engine(DummyEngine.id, DummyEngine)
    assert _ENGINE_REGISTRY["dummy_engine"] is DummyEngine
    compressed, trace = DummyEngine().compress("alpha", 3)
    assert isinstance(compressed, CompressedMemory)
    assert isinstance(trace, CompressionTrace)
