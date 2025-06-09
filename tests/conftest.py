import sys
from pathlib import Path
from typing import Any

import pytest

from compact_memory.embedding_pipeline import MockEncoder


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def patch_hf_auto(monkeypatch: pytest.MonkeyPatch) -> Any:
    class DummyTokenizer:
        model_max_length = 120

        def __call__(
            self, text: str, return_tensors=None, truncation=None, max_length=None
        ):
            ids = list(range(len(text.split())))
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]
            return {"input_ids": [ids]}

        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            if isinstance(ids, list):
                return " ".join(f"t{i}" for i in ids)
            return "tok"

    class DummyModel:
        def __init__(self, *a, **k) -> None:
            self.config = type("cfg", (), {"n_positions": 120})()

        def generate(self, **kw: Any):
            return [[0]]

    monkeypatch.setattr(
        "compact_memory.local_llm.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "compact_memory.local_llm.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel(),
    )
    yield


@pytest.fixture(autouse=True)
def patch_embedding_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    from compact_memory import embedding_pipeline as ep

    original = ep._load_model
    enc = MockEncoder()
    monkeypatch.setattr(ep, "_load_model", lambda *a, **k: enc)
    yield original


@pytest.fixture
def sample_vector_store() -> Any:
    """Return a vector store populated with a single known prototype."""
    from compact_memory.vector_store import InMemoryVectorStore
    from compact_memory.models import BeliefPrototype, RawMemory

    enc = MockEncoder()
    store = InMemoryVectorStore(embedding_dim=enc.dim)
    proto = BeliefPrototype(prototype_id="p1", vector_row_index=0, summary_text="hello")
    vec = enc.encode("hello")
    store.add_prototype(proto, vec)
    store.add_memory(
        RawMemory(
            memory_id="m1",
            raw_text_hash="hash",
            raw_text="hello",
            embedding=vec.tolist(),
        )
    )
    return store
