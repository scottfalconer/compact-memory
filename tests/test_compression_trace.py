import pytest
from compact_memory.engines.base import CompressionTrace


def test_add_step():
    trace = CompressionTrace(
        engine_name="test", strategy_params={}, input_summary={}, output_summary={}
    )
    trace.add_step("chunk_text", {"n": 3})
    assert trace.steps[0]["type"] == "chunk_text"
    assert trace.steps[0]["details"] == {"n": 3}
