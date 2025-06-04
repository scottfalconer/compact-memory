import sys
from pathlib import Path
from typing import Any

import pytest

from compact_memory.embedding_pipeline import MockEncoder


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def patch_hf_auto(monkeypatch: pytest.MonkeyPatch) -> Any:
    yield


@pytest.fixture(autouse=True)
def patch_embedding_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    from compact_memory import embedding_pipeline as ep

    original = ep._load_model
    enc = MockEncoder()
    monkeypatch.setattr(ep, "_load_model", lambda *a, **k: enc)
    yield original
