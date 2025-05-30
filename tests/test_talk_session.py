import pytest

from gist_memory.talk_session import TalkSessionManager
from gist_memory.agent import Agent
from gist_memory.json_npy_store import JsonNpyVectorStore
from gist_memory.embedding_pipeline import MockEncoder
from gist_memory.chunker import SentenceWindowChunker


@pytest.fixture(autouse=True)
def use_mock_encoder(monkeypatch):
    enc = MockEncoder()
    monkeypatch.setattr("gist_memory.embedding_pipeline._load_model", lambda *a, **k: enc)
    yield


def _create_brain(path):
    store = JsonNpyVectorStore(
        path=str(path), embedding_model="mock", embedding_dim=MockEncoder.dim
    )
    agent = Agent(store, chunker=SentenceWindowChunker())
    # avoid conflict flagging noise in tests
    agent._conflicts = type("Dummy", (), {"check_pair": lambda *a, **k: None})()
    agent.add_memory("alpha")
    store.save()


def test_create_and_end_session(tmp_path):
    brain1 = tmp_path / "b1"
    brain2 = tmp_path / "b2"
    _create_brain(brain1)
    _create_brain(brain2)

    mgr = TalkSessionManager()
    sid = mgr.create_session([brain1, brain2])

    assert sid in mgr._sessions
    session = mgr.get_session(sid)
    assert len(session.agents) == 2

    mgr.post_message(sid, "user", "hello")
    assert session.log[-1] == ("user", "hello")

    mgr.end_session(sid)
    assert sid not in mgr._sessions


def test_unique_ids(tmp_path):
    brain = tmp_path / "b"
    _create_brain(brain)
    mgr = TalkSessionManager()
    sid1 = mgr.create_session([brain])
    sid2 = mgr.create_session([brain])
    assert sid1 != sid2


def test_invite_and_kick(tmp_path):
    brain1 = tmp_path / "b1"
    brain2 = tmp_path / "b2"
    brain3 = tmp_path / "b3"
    _create_brain(brain1)
    _create_brain(brain2)
    _create_brain(brain3)

    mgr = TalkSessionManager()
    sid = mgr.create_session([brain1, brain2])
    mgr.post_message(sid, "user", "hello")

    mgr.invite_brain(sid, brain3)
    session = mgr.get_session(sid)
    assert len(session.agents) == 3
    texts = [m.raw_text for m in session.agents[str(brain3)].store.memories]
    assert "hello" in texts

    mgr.kick_brain(sid, str(brain2))
    assert str(brain2) not in session.agents
