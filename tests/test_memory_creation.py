from gist_memory.memory_creation import (
    IdentityMemoryCreator,
    ExtractiveSummaryCreator,
    ChunkMemoryCreator,
    LLMSummaryCreator,
)


def test_identity_memory_creator():
    creator = IdentityMemoryCreator()
    text = "hello world"
    assert creator.create(text) == text


def test_extractive_summary_creator():
    creator = ExtractiveSummaryCreator(max_words=3)
    text = "one two three four five"
    assert creator.create(text) == "one two three"


def test_chunk_memory_creator():
    creator = ChunkMemoryCreator(chunk_size=2)
    text = "one two three four"
    chunks = creator.create_all(text)
    assert chunks == ["one two", "three four"]


def test_llm_summary_creator(monkeypatch):
    class Dummy:
        @staticmethod
        def create(*args, **kwargs):
            return {"choices": [{"message": {"content": "summary"}}]}
    import openai
    monkeypatch.setattr(openai.ChatCompletion, "create", Dummy.create)
    creator = LLMSummaryCreator(model="dummy")
    assert creator.create("text") == "summary"
