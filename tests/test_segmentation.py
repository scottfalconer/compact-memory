from pathlib import Path

from gist_memory.segmentation import agentic_split


def test_agentic_split_long_text():
    text = Path(Path(__file__).parent / "data" / "constitution.txt").read_text()
    chunks = agentic_split(text, max_tokens=40)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 80 for c in chunks)
