from gist_memory.local_llm import LocalChatModel


def test_local_chat_model(monkeypatch):
    class DummyTokenizer:
        def __init__(self, *a, **k):
            pass

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [0]}

        def decode(self, ids, skip_special_tokens=True):
            return "prompt response"

    class DummyModel:
        def __init__(self, *a, **k):
            self.config = type("cfg", (), {"n_positions": 1024})()

        def generate(self, **kw):
            return [[0]]

    monkeypatch.setattr(
        "gist_memory.local_llm.AutoTokenizer.from_pretrained", lambda *a, **k: DummyTokenizer()
    )
    monkeypatch.setattr(
        "gist_memory.local_llm.AutoModelForCausalLM.from_pretrained", lambda *a, **k: DummyModel()
    )
    model = LocalChatModel()
    reply = model.reply("prompt")
    assert reply == "response"
