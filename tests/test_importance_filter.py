from gist_memory.importance_filter import dynamic_importance_filter


def test_dynamic_importance_filter():
    text = "Bob: hi\nuh-huh\nWHO: Bob\nWHEN: 2021\nrandom"
    out = dynamic_importance_filter(text)
    assert "Bob: hi" in out
    assert "WHO: Bob" in out
    assert "WHEN: 2021" in out
    assert "uh-huh" not in out
    assert "random" not in out
