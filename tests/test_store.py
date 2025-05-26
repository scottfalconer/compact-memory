import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist_memory.store import PrototypeStore


def test_add_and_query_memory(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        store = PrototypeStore(path=str(tmp_path))
        mem = store.add_memory("hello world")
        results = store.query("hello", n=1)
        ids = {m.id for m in results}
        assert mem.id in ids
    finally:
        os.chdir(cwd)
