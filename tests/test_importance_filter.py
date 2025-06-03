import pytest
from compact_memory.importance_filter import dynamic_importance_filter


def test_dynamic_importance_filter():
    text = "Bob: hi\nuh-huh\nWHO: Bob\nWHEN: 2021\nrandom"
    out = dynamic_importance_filter(text)
    assert "Bob: hi" in out
    assert "WHO: Bob" in out
    assert "WHEN: 2021" in out
    assert "uh-huh" not in out
    assert "random" not in out


def test_dynamic_importance_filter_spacy():
    spacy = pytest.importorskip("spacy")
    from spacy.language import Language

    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "PERSON", "pattern": "Alice"},
        {"label": "DATE", "pattern": "2042"},
    ])

    text = "Random\nAlice met Bob\nThe year is 2042\n"
    out = dynamic_importance_filter(text, nlp=nlp)
    assert "Alice met Bob" in out
    assert "2042" in out
    assert "Random" not in out
