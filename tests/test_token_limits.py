import json
import pytest
from gist_memory.local_llm import LocalChatModel
from gist_memory.embedding_pipeline import MockEncoder


def _setup_encoder(monkeypatch):
    class DummyEncoder(MockEncoder):
        def get_sentence_embedding_dimension(self):
            return self.dim

    enc = DummyEncoder()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc
    )


class DummyTokenizer:
    def __init__(self, *a, **k):
        pass

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        ids = list(range(len(text.split())))
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return {"input_ids": [ids]}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list):
            return " ".join(f"t{i}" for i in ids)
        return "tok"


class DummyModel:
    def __init__(self, *a, **k):
        # context window large enough to leave small budget for input
        self.config = type("cfg", (), {"n_positions": 120})()

    def generate(self, **kw):
        DummyModel.generated = kw["input_ids"][0]
        return [[0]]


class NoLenModel(DummyModel):
    def __init__(self, *a, **k):
        self.config = type("cfg", (), {})()


@pytest.fixture(autouse=True)
def dummy_llm(monkeypatch):
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    yield


def test_reply_truncates_to_limit(monkeypatch):
    _setup_encoder(monkeypatch)

    called = {}

    def filt(text):
        called["text"] = text
        return text

    monkeypatch.setattr("gist_memory.local_llm.dynamic_importance_filter", filt)

    model = LocalChatModel(max_new_tokens=100)
    long_prompt = " ".join(f"w{i}" for i in range(50))
    model.reply(long_prompt)
    max_len = model.model.config.n_positions - model.max_new_tokens
    assert len(DummyModel.generated) <= max_len
    assert "text" in called


def test_cli_talk_prompt_respects_limit(tmp_path, monkeypatch):
    _setup_encoder(monkeypatch)
    from gist_memory.cli import init, add, talk

    init.callback = init.callback if hasattr(init, 'callback') else init
    init(directory=str(tmp_path))

    for i in range(15):
        text = " ".join(f"m{i}_{j}" for j in range(20))
        add(agent_name=str(tmp_path), text=text, file=None, source_id=None, actor=None, dry_run=False)

    captured = {}

    monkeypatch.setattr(
        "gist_memory.local_llm.dynamic_importance_filter", lambda text: "flt"
    )

    def capture_generate(**kw):
        captured["ids"] = kw["input_ids"][0]
        return [[0]]

    monkeypatch.setattr(DummyModel, "generate", capture_generate)

    talk(agent_name=str(tmp_path), message="hi?", model_name="distilgpt2")
    tokens = len(captured["ids"])
    max_len = DummyModel().config.n_positions - LocalChatModel().max_new_tokens
    assert tokens <= max_len



def test_context_length_uses_tokenizer_when_config_missing(monkeypatch):
    _setup_encoder(monkeypatch)

    class Tok(DummyTokenizer):
        model_max_length = 150

    monkeypatch.setattr(
        "gist_memory.local_llm.AutoTokenizer.from_pretrained", lambda *a, **k: Tok()
    )
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: NoLenModel(),
    )

    model = LocalChatModel(max_new_tokens=100)
    long_prompt = " ".join(f"w{i}" for i in range(50))
    model.reply(long_prompt)
    assert len(NoLenModel.generated) <= 50
