import json
from dataclasses import asdict

from compact_memory.engines.base import CompressionTrace, BaseCompressionEngine


def test_trace_serialization_round_trip():
    trace = CompressionTrace(
        engine_name="test", strategy_params={"p": 1}, input_summary={"in": 1}
    )
    trace.steps.append({"type": "do"})
    trace.output_summary = {"out": 1}

    data = json.loads(json.dumps(asdict(trace)))
    assert data["engine_name"] == "test"
    assert data["steps"][0]["type"] == "do"
    assert data["output_summary"]["out"] == 1


def test_base_engine_trace_contents(patch_embedding_model):
    engine = BaseCompressionEngine()
    result = engine.compress("hello world", budget=5)
    trace = result.trace
    assert trace is not None
    assert trace.engine_name == "base_truncate"
    assert trace.output_summary["compressed_length"] == len(result.text)
    assert trace.steps[0]["type"] == "truncate"
