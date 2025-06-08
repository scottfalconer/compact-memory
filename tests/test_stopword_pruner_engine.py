from compact_memory.engines.stopword_pruner_engine import StopwordPrunerEngine


def test_stopword_pruner_basic():
    engine = StopwordPrunerEngine()
    text = "This is actually a very simple example, you know, just a test."
    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()
    assert "actually" not in out
    assert "is" not in out.split()
    assert trace.engine_name == "stopword_pruner"


def test_stopword_pruner_budget_respected():
    engine = StopwordPrunerEngine()
    text = "word " * 30
    compressed, _ = engine.compress(text, llm_token_budget=5)
    assert len(compressed.text.split()) <= 5
