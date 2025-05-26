from gist_memory.memory_creation import (
    IdentityMemoryCreator,
    ExtractiveSummaryCreator,
)


def test_identity_memory_creator():
    creator = IdentityMemoryCreator()
    text = "hello world"
    assert creator.create(text) == text


def test_extractive_summary_creator():
    creator = ExtractiveSummaryCreator(max_words=3)
    text = "one two three four five"
    assert creator.create(text) == "one two three"
