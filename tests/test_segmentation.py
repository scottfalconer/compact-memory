from pathlib import Path

from gist_memory.segmentation import agentic_split, _sentences


def test_agentic_split_long_text():
    text = Path(Path(__file__).parent / "data" / "constitution.txt").read_text()
    chunks = agentic_split(text, max_tokens=40)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 80 for c in chunks)


def test_spacy_sentence_segmentation_abbreviation():
    text = "Dr. Smith arrived at 5 p.m. He said hello. Goodbye."
    sents = _sentences(text)
    assert len(sents) == 3
