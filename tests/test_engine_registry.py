from compact_memory.engines.registry import (
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
        # No __init__ needed if it just uses base

        def compress(self, text_or_chunks, llm_token_budget: int | None, **kwargs) -> CompressedMemory:
            if isinstance(text_or_chunks, list):
                text_input = "".join(text_or_chunks)
            else:
                text_input = str(text_or_chunks)

            # Ensure budget is an int if not None for slicing
            effective_budget = len(text_input)
            if llm_token_budget is not None:
                effective_budget = llm_token_budget

            compressed_text = text_input[:effective_budget]

            # Create CompressedMemory and then attach trace
            cm = CompressedMemory(text=compressed_text, engine_id=self.id)

            # Create and attach trace
            trace = CompressionTrace(
                engine_name=self.id,
                strategy_params={"llm_token_budget": llm_token_budget},
                input_summary={"original_length": len(text_input)},
                output_summary={"compressed_length": len(compressed_text)},
                final_compressed_object_preview=compressed_text[:50]
            )
            cm.trace = trace
            # self.config would be from BaseCompressionEngine, can be left as default
            # cm.engine_config = self.config
            return cm

    register_compression_engine(DummyEngine.id, DummyEngine)
    assert _ENGINE_REGISTRY["dummy_engine"] is DummyEngine

    result = DummyEngine().compress("alpha", 3)
    assert isinstance(result, CompressedMemory)
    assert result.text == "alp" # Given "alpha" and budget 3
    assert result.engine_id == DummyEngine.id
    assert result.trace is not None
    assert result.trace.engine_name == DummyEngine.id
    assert result.trace.input_summary == {"original_length": 5} # len("alpha")
    assert result.trace.output_summary == {"compressed_length": 3} # len("alp")
    assert result.trace.strategy_params == {"llm_token_budget": 3}
