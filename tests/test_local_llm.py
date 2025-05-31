import pytest
from gist_memory.local_llm import LocalChatModel


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
        "gist_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    model = LocalChatModel()
    reply = model.reply("prompt")
    assert reply == "response"


def test_prepare_prompt(monkeypatch, tmp_path):
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
        "gist_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    from gist_memory.json_npy_store import JsonNpyVectorStore
    from gist_memory.models import BeliefPrototype
    from gist_memory.agent import Agent
    from gist_memory.embedding_pipeline import MockEncoder

    enc = MockEncoder()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )

    store = JsonNpyVectorStore(
        path=str(tmp_path), embedding_model="mock", embedding_dim=enc.dim
    )
    proto = BeliefPrototype(
        prototype_id="p1",
        vector_row_index=0,
        summary_text="summary",
        strength=1.0,
        confidence=1.0,
    )
    store.add_prototype(proto, enc.encode("x"))
    agent = Agent(store)

    model = LocalChatModel()
    prepared = model.prepare_prompt(agent, "prompt")
    assert "summary" in prepared or prepared == "prompt"


def test_local_chat_model_failure(monkeypatch):
    def err(*a, **k):
        raise OSError("missing")

    monkeypatch.setattr(
        "gist_memory.local_llm.AutoTokenizer.from_pretrained", err
    )
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoModelForCausalLM.from_pretrained", err
    )

    model = LocalChatModel(model_name="foo")
    with pytest.raises(RuntimeError) as exc:
        model.load_model()
    msg = str(exc.value)
    assert "download-chat-model" in msg
    assert "foo" in msg
