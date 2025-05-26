from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import chromadb
import numpy as np

from .embedder import Embedder, RandomEmbedder


def default_chroma_client() -> chromadb.Client:
    path = os.path.join(os.getcwd(), "gist_memory_db")
    client = chromadb.PersistentClient(path)
    return client


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
    def __init__(
        self,
        client: Optional[chromadb.Client] = None,
        threshold: float = 0.4,
        embedder: Embedder | None = None,
    ):
        self.client = client or default_chroma_client()
        self.base_threshold = threshold
        self.threshold = threshold
        self.proto_collection = self.client.get_or_create_collection("prototypes")
        self.memory_collection = self.client.get_or_create_collection("memories")
        self.embedder = embedder or RandomEmbedder()

    def add_memory(self, text: str) -> Memory:
        embed = np.array(self.embedder.embed(text))
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
        self._adapt_threshold()
        return Memory(id=mid, text=text, prototype_id=pid)

    def query(self, text: str, n: int = 5) -> List[Memory]:
        embed = np.array(self.embedder.embed(text))
        proto_res = self.proto_collection.query(query_embeddings=[embed], n_results=n)
        if not proto_res["ids"]:
            return []

        proto_ids = [pid for pid in proto_res["ids"][0]]
        scored: list[tuple[float, Memory]] = []
        for pid in proto_ids:
            mem_res = self.memory_collection.get(
                where={"prototype_id": pid}, include=["embeddings", "documents"]
            )
            for mid, doc, memb in zip(
                mem_res["ids"], mem_res["documents"], mem_res["embeddings"]
            ):
                dist = float(np.linalg.norm(embed - np.array(memb)))
                scored.append((dist, Memory(id=mid, text=doc, prototype_id=pid)))

        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored[:n]]

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

    def _adapt_threshold(self) -> None:
        """Simple heuristic to adjust the prototype assignment threshold."""
        count = self.proto_collection.count()
        if count <= 1:
            self.threshold = self.base_threshold
        else:
            self.threshold = max(0.05, self.base_threshold / np.sqrt(count))
