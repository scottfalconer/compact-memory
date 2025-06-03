from gist_memory_rationale_episode_strategy import (
    Episode,
    Decision,
    EpisodeStorage,
    RationaleEpisodeManager,
)


def test_episode_roundtrip():
    ep = Episode(
        summary_gist="test",
        tags=["a"],
        decisions=[Decision(step_id="1", step_summary="s", rationale="r")],
    )
    data = ep.to_dict()
    ep2 = Episode.from_dict(data)
    assert ep2.summary_gist == "test"
    assert ep2.tags == ["a"]
    assert ep2.decisions[0].rationale == "r"


def test_manager_finalize_creates_file(tmp_path, monkeypatch):
    class DummyModel:
        def encode(self, text, **kw):
            import numpy as np

            return np.array([0.0, 0.0], dtype=np.float32)

        def get_sentence_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: DummyModel()
    )
    import gist_memory.embedding_pipeline as ep

    ep._embed_cached.cache_clear()
    store = EpisodeStorage(tmp_path)
    mgr = RationaleEpisodeManager(store)
    mgr.store_step("1", "summary", "because", importance=0.8)
    mgr.finalize_episode("episode one", tags=["x"])
    assert (tmp_path / "episodes.jsonl").exists()
    lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_retrieve_returns_episode(tmp_path, monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def encode(self, text, **kw):
            import numpy as np

            if isinstance(text, list):
                arr = [
                    (
                        np.array([1.0, 0.0], dtype=np.float32)
                        if "one" in t
                        else np.array([0.0, 1.0], dtype=np.float32)
                    )
                    for t in text
                ]
                return np.stack(arr)
            return (
                np.array([1.0, 0.0], dtype=np.float32)
                if "one" in text
                else np.array([0.0, 1.0], dtype=np.float32)
            )

        def get_sentence_embedding_dimension(self):
            return 2

    dummy = DummyModel()
    monkeypatch.setattr(
        "gist_memory.embedding_pipeline._load_model", lambda *a, **k: dummy
    )
    import gist_memory.embedding_pipeline as ep

    ep._embed_cached.cache_clear()

    store = EpisodeStorage(tmp_path)
    mgr = RationaleEpisodeManager(store)
    mgr.store_step("1", "sum", "why")
    mgr.finalize_episode("episode one", tags=["a"])
    mgr.store_step("2", "sum", "why")
    mgr.finalize_episode("episode two", tags=["b"])

    results = mgr.retrieve("episode one")
    assert results
    assert results[0].summary_gist.startswith("episode one")
