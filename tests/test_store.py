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


def test_summarize_prototype():
    client = chromadb.EphemeralClient()
    store = PrototypeStore(client=client)
    mem1 = store.add_memory("alpha bravo charlie")
    summary = store.summarize_prototype(mem1.prototype_id, max_words=2)
    assert summary == "alpha bravo"


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


def test_ingest_long_text_end_state():
    text = """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."""

    client = chromadb.EphemeralClient()
    store = PrototypeStore(client=client)
    mem = store.add_memory(text)

    all_mems = store.dump_memories()
    assert any(m.id == mem.id and m.text == text for m in all_mems)
