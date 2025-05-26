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
