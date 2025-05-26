from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import chromadb
import numpy as np

from .embedder import RandomEmbedder


def default_chroma_client(path: Optional[str] = None) -> chromadb.Client:
    """Create a Chroma client using PATH or the CWD."""
    db_path = path or os.path.join(os.getcwd(), "gist_memory_db")
    return chromadb.PersistentClient(db_path)


@dataclass
class Memory:
    id: str
    text: str
    prototype_id: str


@dataclass
class Prototype:
    id: str
    embedding: np.ndarray


class PrototypeStore:
    def __init__(self, client: Optional[chromadb.Client] = None, *, path: Optional[str] = None, threshold: float = 0.4):
        """Create a prototype store.

        Parameters
        ----------
        client: chromadb.Client | None
            Pre-initialised Chroma client. If ``None`` a persistent client will
            be created in ``path``.
        path: str | None
            Directory to store the persistent database. Defaults to
            ``gist_memory_db`` in the current working directory.
        threshold: float
            Distance threshold for assigning memories to prototypes.
        """
        self.client = client or default_chroma_client(path)
        self.threshold = threshold
        self.proto_collection = self.client.get_or_create_collection("prototypes")
        self.memory_collection = self.client.get_or_create_collection("memories")
        # Using a local random embedder to avoid network downloads in this prototype
        self.emb_func = RandomEmbedder().embed

    def add_memory(self, text: str) -> Memory:
        embed = np.array(self.emb_func(text))
        # find nearest prototype
        if self.proto_collection.count() > 0:
            res = self.proto_collection.query(query_embeddings=[embed], n_results=1)
            pid = res["ids"][0][0]
            dist = res["distances"][0][0]
            if dist > self.threshold:
                pid = self._create_prototype(embed)
            else:
                self._update_prototype(pid, embed)
        else:
            pid = self._create_prototype(embed)
        mid = str(uuid.uuid4())
        self.memory_collection.add(ids=[mid], embeddings=[embed], metadatas=[{"prototype_id": pid}], documents=[text])
        return Memory(id=mid, text=text, prototype_id=pid)

    def query(self, text: str, n: int = 5) -> List[Memory]:
        embed = np.array(self.emb_func(text))
        res = self.proto_collection.query(query_embeddings=[embed], n_results=n)
        if not res["ids"]:
            return []
        proto_ids = [pid for pid in res["ids"][0]]
        results = []
        for pid in proto_ids:
            mem_res = self.memory_collection.get(where={"prototype_id": pid})
            for mid, doc in zip(mem_res["ids"], mem_res["documents"]):
                results.append(Memory(id=mid, text=doc, prototype_id=pid))
        return results

    def _create_prototype(self, embed: np.ndarray) -> str:
        pid = str(uuid.uuid4())
        self.proto_collection.add(ids=[pid], embeddings=[embed])
        return pid

    def _update_prototype(self, pid: str, embed: np.ndarray, alpha: float = 0.1):
        proto = self.proto_collection.get(ids=[pid])
        if not proto["ids"]:
            self.proto_collection.add(ids=[pid], embeddings=[embed])
            return
        old = np.array(proto["embeddings"][0])
        new = (1 - alpha) * old + alpha * embed
        self.proto_collection.update(ids=[pid], embeddings=[new])
