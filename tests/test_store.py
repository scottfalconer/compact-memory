import chromadb

from gist_memory.store import PrototypeStore


def test_add_and_query_memory():
    client = chromadb.EphemeralClient()
    store = PrototypeStore(client=client)
    mem = store.add_memory("hello world")
    assert mem.text == "hello world"
    assert mem.prototype_id

    results = store.query("hello", n=1)
    assert results
    assert results[0].text == "hello world"
