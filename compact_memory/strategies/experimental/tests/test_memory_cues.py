from compact_memory.strategies.experimental.prototype_system_utils import (
    MemoryCueRenderer,
)


def test_memory_cue_renderer_basic():
    renderer = MemoryCueRenderer(max_words=2)
    cues = renderer.render(["Budget 2025 planning", "Alpha Beta Gamma"])
    assert "<MEM id=Budget_2025 />" in cues
    assert "<MEM id=Alpha_Beta />" in cues
