from compact_memory.engines.stopword_pruner_engine import StopwordPrunerEngine
import nltk # For checking NLTK's list in the test if needed for clarity, though engine handles it.

def test_stopword_pruner_basic():
    engine = StopwordPrunerEngine()
    text = "This is actually a very simple example, you know, just a test."
    # Ensure NLTK stopwords are available for the engine to pick up if spaCy doesn't mark a word as stop.
    # This command should be run in the environment setup if not already:
    # import nltk; nltk.download('stopwords', quiet=True)

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    # Check specific words are removed or kept, accounting for NLTK's list
    assert "actually" not in out, f"'actually' (filler) should be removed, output: {out}"
    assert "is" not in out.split(), f"'is' (stopword) should be removed, output: {out}"
    assert "this" not in out.split(), f"'this' (stopword) should be removed, output: {out}"
    assert "a" not in out.split(), f"'a' (stopword) should be removed, output: {out}"
    assert "you" not in out.split(), f"'you' (stopword by spaCy) should be removed, output: {out}"

    # "very" and "just" are stopwords in NLTK's default English list
    # and the engine uses NLTK list if token.is_stop is False.
    assert "very" not in out.split(), f"'very' (stopword by NLTK) should be removed, output: {out}"
    assert "just" not in out.split(), f"'just' (stopword by NLTK) should be removed, output: {out}"

    assert "know" in out, f"'know' should be present, output: {out}"
    assert "simple" in out, f"'simple' should be present, output: {out}"
    assert "example" in out, f"'example' should be present, output: {out}"
    assert "test" in out, f"'test' should be present, output: {out}"


    # Check punctuation removal
    assert "," not in out, f"',' should be removed, output: {out}"
    assert "." not in out, f"'.' (from 'test.') should be removed, output: {out}"

    # Assert final output string based on expected token processing with NLTK stopwords
    assert out == "simple example know test", f"Output string mismatch. Expected 'simple example know test', got '{out}'"

    assert trace.engine_name == "stopword_pruner", "Trace engine name mismatch"

    # Verify trace details for removed counts
    removed_stopwords_count = 0
    removed_fillers_count = 0
    for step in trace.steps:
        if step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]
        elif step["type"] == "remove_fillers":
            removed_fillers_count = step["removed"]

    # Expected counts for "This is actually a very simple example, you know, just a test."
    # using spaCy 'en_core_web_sm' AND NLTK's English stopwords:
    # Stopwords: "This"(spaCy), "is"(spaCy), "a"(spaCy), "very"(NLTK), "you"(spaCy), "just"(NLTK), "a"(spaCy) (7)
    # Fillers: "actually" (1)
    assert removed_stopwords_count == 7, f"Expected 7 stopwords removed, got {removed_stopwords_count}. Trace steps: {trace.steps}"
    assert removed_fillers_count == 1, f"Expected 1 filler removed, got {removed_fillers_count}. Trace steps: {trace.steps}"

def test_stopword_pruner_budget_respected():
    engine = StopwordPrunerEngine()
    text = "word " * 30
    compressed, _ = engine.compress(text, llm_token_budget=5)
    output_tokens = compressed.text.split()
    assert len(output_tokens) <= 5, f"Expected <= 5 tokens, got {len(output_tokens)}: '{compressed.text}'"
    for token in output_tokens:
        assert token == "word", f"Expected token 'word', got '{token}'"

def test_stopword_pruner_min_word_length():
    engine = StopwordPrunerEngine(config={"min_word_length": 4})
    text = "A cat ran fast to the big tree."
    # SpaCy/NLTK stopwords: "A", "to", "the" (3 stopwords)
    # Words shorter than 4 chars (and not stopwords/fillers): "cat" (3), "ran" (3), "big" (3) (3 short words)
    # Kept words: "fast" (4), "tree" (4)
    # Expected output: "fast tree"

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "fast tree", f"Output string mismatch. Expected 'fast tree', got '{out}'"

    removed_short_count = 0
    removed_stopwords_count = 0
    for step in trace.steps:
        if step["type"] == "remove_short":
            removed_short_count = step["removed"]
        elif step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]

    assert removed_short_count == 3, f"Expected 3 short words removed, got {removed_short_count}. Trace: {trace.steps}"
    # "A" is a stopword by spaCy. "to", "the" are also typical stopwords.
    # NLTK's list is also active in the engine's logic.
    # "A" (spaCy stop), "to" (spaCy stop), "the" (spaCy stop)
    assert removed_stopwords_count == 3, f"Expected 3 stopwords removed, got {removed_stopwords_count}. Trace: {trace.steps}"

def test_stopword_pruner_remove_fillers_false():
    engine = StopwordPrunerEngine(config={"remove_fillers": False})
    text = "This is, like, a test, um, to see."
    # Fillers to be kept: "like", "um"
    # Stopwords to be removed: "This", "is", "a", "to", "see" (spaCy marks "see" as stopword) (5)
    # Punctuation to be removed: ",", ","
    # Other words to keep: "test"
    # Expected output: "like test um"

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "like test um", f"Output string mismatch. Expected 'like test um', got '{out}'"

    removed_fillers_count = 0
    removed_stopwords_count = 0
    for step in trace.steps:
        if step["type"] == "remove_fillers":
            removed_fillers_count = step["removed"]
        elif step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]

    assert removed_fillers_count == 0, f"Expected 0 fillers removed, got {removed_fillers_count}. Trace: {trace.steps}"
    assert removed_stopwords_count == 5, f"Expected 5 stopwords removed (This, is, a, to, see), got {removed_stopwords_count}. Trace: {trace.steps}"

def test_stopword_pruner_remove_duplicate_words_true():
    engine = StopwordPrunerEngine(config={"remove_duplicates": True})
    # "Yes" and "agree" are not default NLTK/spaCy stopwords.
    # "I", "with", "you" are stopwords.
    text = "Yes yes I I agree agree with you you."
    # Expected processing:
    # "Yes" (keep)
    # "yes" (duplicate of prev_token_lower "yes", remove, dup_count=1)
    # "I" (stopword, remove, stop_count=1)
    # "I" (stopword, remove, stop_count=2)
    # "agree" (keep, prev_token_lower="agree")
    # "agree" (duplicate of prev_token_lower "agree", remove, dup_count=2)
    # "with" (stopword, remove, stop_count=3)
    # "you" (stopword, remove, stop_count=4)
    # "you" (stopword, remove, stop_count=5)
    # "." (punctuation, remove)
    # Output: "Yes agree"

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "yes agree", f"Output string mismatch. Expected 'yes agree', got '{out}'"

    removed_duplicates_count = 0
    removed_stopwords_count = 0
    for step in trace.steps:
        if step["type"] == "remove_duplicates":
            removed_duplicates_count = step["removed"]
        elif step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]

    assert removed_duplicates_count == 2, f"Expected 2 duplicate words removed, got {removed_duplicates_count}. Trace: {trace.steps}"
    assert removed_stopwords_count == 5, f"Expected 5 stopwords removed, got {removed_stopwords_count}. Trace: {trace.steps}"

def test_stopword_pruner_remove_duplicate_sentences_true():
    engine = StopwordPrunerEngine(config={"remove_duplicates": True, "min_word_length": 1})
    text = "Sentence one. Sentence two. Sentence one."
    # Expected processing:
    # Sent 1: "Sentence" (keep), "one" (stopword by spaCy, remove). prev_token_lower = "sentence".
    # Sent 2: "Sentence" (duplicate word of prev "sentence", remove), "two" (stopword by spaCy, remove).
    # Sent 3: "Sentence one." (duplicate sentence, remove tokens "Sentence", "one").
    # Punctuation "." removed from all.
    # Output: "sentence"

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "sentence", f"Output string mismatch. Expected 'sentence', got '{out}'"

    removed_duplicates_count = 0
    removed_stopwords_count = 0
    for step in trace.steps:
        if step["type"] == "remove_duplicates":
            removed_duplicates_count = step["removed"]
        elif step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]

    # Duplicates:
    # 1. Word "Sentence" from "Sentence two." (prev was "Sentence" from "Sentence one.")
    # 2. Tokens "Sentence", "one", "." from the third sentence ("Sentence one.") being a duplicate sentence, matched by _TOKEN_RE.
    # Total = 1 (word) + 3 (sentence tokens) = 4
    assert removed_duplicates_count == 4, f"Expected 4 duplicate tokens removed (1 word, 3 from sent), got {removed_duplicates_count}. Trace: {trace.steps}"
    # Stopwords: "one" (from sent 1), "two" (from sent 2) - spaCy marks them as is_stop.
    assert removed_stopwords_count == 2, f"Expected 2 stopwords removed ('one', 'two'), got {removed_stopwords_count}. Trace: {trace.steps}"

def test_stopword_pruner_remove_duplicates_false():
    # Test with explicit config, though remove_duplicates=False is default.
    engine = StopwordPrunerEngine(config={"remove_duplicates": False, "min_word_length": 1})
    text = "A word word. A sentence. A sentence."
    # Expected:
    # "A" (stopword) -> removed (x3)
    # "word" (kept)
    # "word" (kept, duplicates=False)
    # "sentence" (kept)
    # "sentence" (kept, duplicates=False)
    # Punctuation "." removed.
    # Output: "word word sentence sentence"

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "word word sentence sentence", f"Output string mismatch. Expected 'word word sentence sentence', got '{out}'"

    removed_duplicates_count = 0
    removed_stopwords_count = 0
    for step in trace.steps:
        if step["type"] == "remove_duplicates":
            removed_duplicates_count = step["removed"]
        elif step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]

    assert removed_duplicates_count == 0, f"Expected 0 duplicate words/sentences removed, got {removed_duplicates_count}. Trace: {trace.steps}"
    assert removed_stopwords_count == 3, f"Expected 3 stopwords removed (the three 'A's), got {removed_stopwords_count}. Trace: {trace.steps}"

def test_stopword_pruner_preserve_order_false():
    engine = StopwordPrunerEngine(config={
        "preserve_order": False,
        "min_word_length": 1,
        "remove_duplicates": False # Explicitly show preserve_order=False makes unique
    })
    text = "zebra alpha charlie bravo alpha"
    # No stopwords in this text.
    # output_tokens before sort: ["zebra", "alpha", "charlie", "bravo", "alpha"] (if remove_duplicates=False)
    # Then, sorted(set(output_tokens)) = sorted({"zebra", "alpha", "charlie", "bravo"})
    # = ["alpha", "bravo", "charlie", "zebra"]

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "alpha bravo charlie zebra",         f"Output string mismatch. Expected 'alpha bravo charlie zebra', got '{out}'"
    # Verify no other removals happened unexpectedly
    for step_type in ["stopwords", "fillers", "short", "duplicates"]:
        removed_count = 0
        for step in trace.steps:
            if step["type"] == f"remove_{step_type}":
                removed_count = step["removed"]
        assert removed_count == 0, f"Expected 0 {step_type} removed, got {removed_count} for '{text}'. Trace: {trace.steps}"

def test_stopword_pruner_preserve_order_true_maintains_order():
    engine = StopwordPrunerEngine(config={
        "preserve_order": True,
        "remove_duplicates": False, # Test with duplicates present
        "min_word_length": 1
    })
    text = "zebra alpha charlie bravo alpha"
    # Expected: all tokens kept in original order.

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "zebra alpha charlie bravo alpha",         f"Output string mismatch. Expected 'zebra alpha charlie bravo alpha', got '{out}'"
    # Verify no other removals happened unexpectedly
    for step_type in ["stopwords", "fillers", "short", "duplicates"]: # duplicates should be 0 here by config
        removed_count = 0
        for step in trace.steps:
            if step["type"] == f"remove_{step_type}":
                removed_count = step["removed"]
        assert removed_count == 0, f"Expected 0 {step_type} removed, got {removed_count} for '{text}'. Trace: {trace.steps}"

def test_stopword_pruner_language_spanish():
    # First, ensure NLTK has Spanish stopwords list.
    # This is a check that the test environment is as expected.
    import nltk
    try:
        nltk.corpus.stopwords.words("spanish")
    except OSError: # OSError if 'stopwords' corpus not found or language missing
        nltk.download('stopwords', quiet=True) # Attempt download
        # If it still fails, this test might not be able to run correctly,
        # but the engine itself has a fallback. For this test, we want to see NLTK Spanish work.
        # Re-check after download attempt.
        try:
            nltk.corpus.stopwords.words("spanish")
        except Exception as e:
            # If spanish stopwords are truly unavailable after download attempt,
            # this specific test for Spanish cannot pass as intended.
            # We could skip or expect fallback, but plan is to test Spanish.
            # For now, assume it will be available in a well-set-up test env.
            pass # Let it proceed, engine might fallback.

    engine = StopwordPrunerEngine(config={"stopwords_language": "spanish", "min_word_length": 1})
    text = "el perro come una manzana"
    # "el" and "una" are common Spanish stopwords.
    # "perro", "come", "manzana" are not.
    # en_core_web_sm (default spaCy) won't mark "el", "una" as is_stop.
    # So removal depends on NLTK's Spanish list via _get_stopwords("spanish").

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "perro come manzana",         f"Output string mismatch. Expected 'perro come manzana', got '{out}'"

    removed_stopwords_count = 0
    for step in trace.steps:
        if step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]

    assert removed_stopwords_count == 2,         f"Expected 2 Spanish stopwords removed, got {removed_stopwords_count}. Trace: {trace.steps}"

def test_stopword_pruner_punctuation_whitespace():
    engine = StopwordPrunerEngine(config={"min_word_length": 1})
    text = "  Hello,   world!   How are you? End.  "
    # Expected processing:
    # Leading/trailing/excess internal spaces: handled by tokenization and join.
    # Punctuation: ",", "!", "?", "." -> removed.
    # Stopwords: "How", "are", "you" -> removed (count=3).
    # Kept: "Hello", "world", "End".
    # Output: "hello world end" (after test's lowercasing)

    compressed, trace = engine.compress(text, llm_token_budget=100)
    out = compressed.text.lower()

    assert out == "hello world end",         f"Output string mismatch. Expected 'hello world end', got '{out}'"

    removed_stopwords_count = 0
    removed_fillers_count = 0
    removed_short_count = 0
    removed_duplicates_count = 0

    for step in trace.steps:
        if step["type"] == "remove_stopwords":
            removed_stopwords_count = step["removed"]
        elif step["type"] == "remove_fillers":
            removed_fillers_count = step["removed"]
        elif step["type"] == "remove_short":
            removed_short_count = step["removed"]
        elif step["type"] == "remove_duplicates":
            removed_duplicates_count = step["removed"]

    assert removed_stopwords_count == 3,         f"Expected 3 stopwords removed, got {removed_stopwords_count}. Trace: {trace.steps}"
    assert removed_fillers_count == 0,         f"Expected 0 fillers removed, got {removed_fillers_count}. Trace: {trace.steps}"
    assert removed_short_count == 0,         f"Expected 0 short words removed, got {removed_short_count}. Trace: {trace.steps}"
    assert removed_duplicates_count == 0,         f"Expected 0 duplicates removed, got {removed_duplicates_count}. Trace: {trace.steps}"
