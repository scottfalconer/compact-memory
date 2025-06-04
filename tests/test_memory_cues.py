from contrib.prototype_utils.memory_cues import MemoryCueRenderer


def test_memory_cue_renderer_basic():
    renderer = MemoryCueRenderer(max_words=2)
    cues = renderer.render(["Budget 2025 planning", "Alpha Beta Gamma"])
    assert "<MEM id=Budget_2025 />" in cues
    assert "<MEM id=Alpha_Beta />" in cues
