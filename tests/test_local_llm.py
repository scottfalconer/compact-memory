import pytest
from compact_memory.local_llm import LocalChatModel


def test_local_chat_model(monkeypatch):
    class DummyTokenizer:
        def __init__(self, *a, **k):
            pass

        def __call__(
            self, prompt, return_tensors=None, truncation=None, max_length=None
        ):
            return {"input_ids": [0]}

        def decode(self, ids, skip_special_tokens=True):
            return "prompt response"

    class DummyModel:
        def __init__(self, *a, **k):
            self.config = type("cfg", (), {"n_positions": 1024})()

        def generate(self, **kw):
            return [[0]]

    monkeypatch.setattr(
        "compact_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "compact_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    model = LocalChatModel()
    reply = model.reply("prompt")
    assert reply == "response"


@pytest.mark.skip(reason="prepare_prompt not implemented")
def test_prepare_prompt(monkeypatch):
    class DummyTokenizer:
        def __init__(self, *a, **k):
            pass

        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [list(range(10))]}

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, list):
                return "old" if len(ids) > 5 else "recent"
            return "prompt"

    class DummyModel:
        def __init__(self, *a, **k):
            self.config = type("cfg", (), {"n_positions": 5})()

        def generate(self, **kw):
            return [[0]]

    monkeypatch.setattr(
        "compact_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "compact_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    from compact_memory.models import BeliefPrototype

    class DummyStore:
        def __init__(self) -> None:
            self.prototypes = [
                BeliefPrototype(
                    prototype_id="p1", vector_row_index=0, summary_text="summary"
                )
            ]

        def find_nearest(self, vec, k=3):
            return [("p1", 1.0)]

    class DummyAgent:
        def __init__(self) -> None:
            self.store = DummyStore()

    monkeypatch.setattr(
        "compact_memory.memory_creation.DefaultTemplateBuilder.build",
        lambda self, text, slots: text,
    )
    monkeypatch.setattr(
        "compact_memory.embedding_pipeline.embed_text", lambda text: [0.0]
    )

    model = LocalChatModel()
    prepared = model.prepare_prompt(DummyAgent(), "prompt")
    assert "summary" in prepared or prepared == "prompt"


def test_local_chat_model_failure(monkeypatch):
    def err(*a, **k):
        raise OSError("missing")

    monkeypatch.setattr(
        "compact_memory.local_llm.AutoTokenizer.from_pretrained",
        err,
    )
    monkeypatch.setattr(
        "compact_memory.local_llm.AutoModelForCausalLM.from_pretrained", err
    )

    model = LocalChatModel(model_name="foo")
    with pytest.raises(RuntimeError) as exc:
        model.load_model()
    msg = str(exc.value)
    assert "download-chat-model" in msg
    assert "foo" in msg


def test_provider_interface(monkeypatch):
    class DummyTokenizer:
        def __init__(self, *a, **k):
            pass

        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [0]}

        def decode(self, ids, skip_special_tokens=True):
            return "prompt response"

    class DummyModel:
        def __init__(self, *a, **k):
            self.config = type("cfg", (), {"n_positions": 50})()

        def generate(self, **kw):
            return [[0]]

    monkeypatch.setattr(
        "compact_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "compact_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    model = LocalChatModel()
    budget = model.get_token_budget("tiny-gpt2")
    assert isinstance(budget, int) and budget > 0
    tokens = model.count_tokens("hello", "tiny-gpt2")
    assert tokens == 1
    reply = model.generate_response("prompt", "tiny-gpt2", 10)
    assert reply == "response"
