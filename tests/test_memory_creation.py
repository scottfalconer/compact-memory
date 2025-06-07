from compact_memory.memory_creation import (
    IdentityMemoryCreator,
    ExtractiveSummaryCreator,
    ChunkMemoryCreator,
    LLMSummaryCreator,
    TemplateBuilder,
    DefaultTemplateBuilder,
    register_template_builder,
    _TEMPLATE_REGISTRY,
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


def test_agentic_memory_creator():
    from compact_memory.memory_creation import AgenticMemoryCreator

    text = "A. B. C. D. E."
    creator = AgenticMemoryCreator(max_tokens=2, sim_threshold=0.1)
    chunks = creator.create_all(text)
    assert len(chunks) >= 3


def test_llm_summary_creator(monkeypatch):
    class Dummy:
        @staticmethod
        def create(*args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
            )

    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(openai.chat.completions, "create", Dummy.create)
    creator = LLMSummaryCreator(model="dummy")
    assert creator.create("text") == "summary"


def test_default_template_builder():
    builder = DefaultTemplateBuilder()
    out = builder.build("hello", {"who": "Alice", "why": "testing"})
    assert "hello" in out
    assert "who:Alice" in out
    assert "why:testing" in out


def test_register_template_builder():
    class DummyTemplate(TemplateBuilder):
        id = "dummy"

        def build(self, sentence: str, slots: dict[str, str]) -> str:
            return "dummy"

    register_template_builder(DummyTemplate.id, DummyTemplate)
    assert _TEMPLATE_REGISTRY["dummy"] is DummyTemplate
