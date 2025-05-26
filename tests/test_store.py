import chromadb

from gist_memory.store import PrototypeStore
import numpy as np
import tempfile
import shutil


def test_add_and_query_memory():
    client = chromadb.EphemeralClient()
    store = PrototypeStore(client=client)
    mem = store.add_memory("hello world")
    assert mem.text == "hello world"
    assert mem.prototype_id

    results = store.query("hello", n=1)
    assert results
    assert results[0].text == "hello world"


def test_threshold_adaptation():
    path = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path)
    store = PrototypeStore(client=client, threshold=0.4)
    store.emb_func = lambda t: np.zeros(768)

    store._create_prototype(np.zeros(768))
    store._adapt_threshold()
    assert store.threshold == store.base_threshold

    store._create_prototype(np.ones(768))
    store._adapt_threshold()
    assert store.threshold < store.base_threshold
    shutil.rmtree(path)


def test_query_ranking():
    """Memories should be ranked by distance to the query."""
    class DummyEmbedder:
        def __init__(self):
            z = np.zeros(768)
            a = np.zeros(768); a[:2] = [0.0, 0.0]
            b = np.zeros(768); b[:2] = [1.0, 1.0]
            q = np.zeros(768); q[:2] = [0.0, 0.1]
            self.map = {
                "hello world": a,
                "bye world": b,
                "hello": q,
            }

        def embed(self, text):
            return self.map[text]

    client = chromadb.EphemeralClient()
    embedder = DummyEmbedder()
    store = PrototypeStore(client=client, embedder=embedder)
    store.add_memory("hello world")
    store.add_memory("bye world")

    results = store.query("hello", n=1)
    assert len(results) == 1
    assert results[0].text == "hello world"


def test_decode_prototype():
    client = chromadb.EphemeralClient()
    store = PrototypeStore(client=client)
    mem = store.add_memory("alpha bravo")
    mems = store.decode_prototype(mem.prototype_id, n=1)
    assert mems
    assert mems[0].text == "alpha bravo"


def test_dump_memories():
    client = chromadb.EphemeralClient()
    store = PrototypeStore(client=client)
    mem1 = store.add_memory("foo")
    mem2 = store.add_memory("bar")
    all_mems = store.dump_memories()
    texts = {m.text for m in all_mems}
    assert {"foo", "bar"}.issubset(texts)
    filtered = store.dump_memories(prototype_id=mem1.prototype_id)
    assert any(m.text == "foo" for m in filtered)
