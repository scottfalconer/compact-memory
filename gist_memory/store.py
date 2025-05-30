from __future__ import annotations
from abc import ABC, abstractmethod

import json
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .embedder import Embedder, RandomEmbedder

try:  # optional chroma dependency
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional
    chromadb = None


@dataclass
class Memory:
    id: str
    text: str
    prototype_id: str


@dataclass
class Prototype:
    id: str
    embedding: np.ndarray


class VectorStore(ABC):
    """Abstract vector store interface."""

    def __init__(
        self,
        threshold: float = 0.4,
        *,
        min_threshold: float = 0.05,
        decay_exponent: float = 0.5,
        embedder: Embedder | None = None,
    ) -> None:
        self.base_threshold = threshold
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.decay_exponent = decay_exponent
        self.embedder = embedder or RandomEmbedder()

    @abstractmethod
    def add_memory(self, text: str) -> Memory: ...

    @abstractmethod
    def query(self, text: str, n: int = 5) -> List[Memory]: ...

    @abstractmethod
    def decode_prototype(self, pid: str, n: int = 1) -> List[Memory]: ...

    @abstractmethod
    def summarize_prototype(self, pid: str, max_words: int = 50) -> str | None: ...

    @abstractmethod
    def dump_memories(self, prototype_id: str | None = None) -> List[Memory]: ...

    @abstractmethod
    def prototype_count(self) -> int: ...


class JSONVectorStore(VectorStore):
    """Simple JSON/NPY backed store."""

    def __init__(
        self,
        path: Optional[str] = None,
        threshold: float = 0.4,
        *,
        min_threshold: float = 0.05,
        decay_exponent: float = 0.5,
        embedder: Embedder | None = None,
    ) -> None:
        super().__init__(
            threshold,
            min_threshold=min_threshold,
            decay_exponent=decay_exponent,
            embedder=embedder,
        )
        self.path = path  # None -> ephemeral
        self.prototypes: list[str] = []
        self.proto_embeds: list[np.ndarray] = []
        self.memories: list[Memory] = []
        self.mem_embeds: list[np.ndarray] = []
        if path:
            self._load()

    # persistence helpers -------------------------------------------------
    def _proto_json_path(self) -> str:
        assert self.path
        return os.path.join(self.path, "prototypes.json")

    def _proto_npy_path(self) -> str:
        assert self.path
        return os.path.join(self.path, "prototypes.npy")

    def _mem_json_path(self) -> str:
        assert self.path
        return os.path.join(self.path, "memories.json")

    def _mem_npy_path(self) -> str:
        assert self.path
        return os.path.join(self.path, "memories.npy")

    def _load(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        if os.path.exists(self._proto_json_path()):
            with open(self._proto_json_path(), "r") as f:
                data = json.load(f)
                self.prototypes = data.get("ids", [])
            if os.path.exists(self._proto_npy_path()):
                arr = np.load(self._proto_npy_path())
                self.proto_embeds = [arr[i] for i in range(len(self.prototypes))]
        if os.path.exists(self._mem_json_path()):
            with open(self._mem_json_path(), "r") as f:
                data = json.load(f)
                ids = data.get("ids", [])
                texts = data.get("texts", [])
                pids = data.get("prototype_ids", [])
                self.memories = [
                    Memory(id=i, text=t, prototype_id=p)
                    for i, t, p in zip(ids, texts, pids)
                ]
            if os.path.exists(self._mem_npy_path()):
                arr = np.load(self._mem_npy_path())
                self.mem_embeds = [arr[i] for i in range(len(self.memories))]

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(self.path, exist_ok=True)
        with open(self._proto_json_path(), "w") as f:
            json.dump({"ids": self.prototypes}, f)
        if self.proto_embeds:
            np.save(self._proto_npy_path(), np.stack(self.proto_embeds))
        with open(self._mem_json_path(), "w") as f:
            json.dump(
                {
                    "ids": [m.id for m in self.memories],
                    "texts": [m.text for m in self.memories],
                    "prototype_ids": [m.prototype_id for m in self.memories],
                },
                f,
            )
        if self.mem_embeds:
            np.save(self._mem_npy_path(), np.stack(self.mem_embeds))

    # ---------------------------------------------------------------------
    def _create_prototype(self, embed: np.ndarray) -> str:
        pid = str(uuid.uuid4())
        self.prototypes.append(pid)
        self.proto_embeds.append(embed)
        return pid

    def _update_prototype(
        self, pid: str, embed: np.ndarray, alpha: float = 0.1
    ) -> None:
        if pid not in self.prototypes:
            self._create_prototype(embed)
            return
        idx = self.prototypes.index(pid)
        old = self.proto_embeds[idx]
        self.proto_embeds[idx] = (1 - alpha) * old + alpha * embed

    def _adapt_threshold(self) -> None:
        count = len(self.prototypes)
        if count <= 1:
            self.threshold = self.base_threshold
        else:
            adjusted = self.base_threshold / (count**self.decay_exponent)
            self.threshold = max(self.min_threshold, adjusted)

    def prototype_count(self) -> int:
        return len(self.prototypes)

    # public API ----------------------------------------------------------
    def add_memory(self, text: str) -> Memory:
        embed = np.array(self.embedder.embed(text))
        if self.prototypes:
            arr = np.stack(self.proto_embeds)
            dists = np.linalg.norm(arr - embed, axis=1)
            idx = int(dists.argmin())
            dist = float(dists[idx])
            if dist > self.threshold:
                pid = self._create_prototype(embed)
            else:
                pid = self.prototypes[idx]
                self._update_prototype(pid, embed)
        else:
            pid = self._create_prototype(embed)
        mid = str(uuid.uuid4())
        self.memories.append(Memory(id=mid, text=text, prototype_id=pid))
        self.mem_embeds.append(embed)
        self._adapt_threshold()
        self._save()
        return Memory(id=mid, text=text, prototype_id=pid)

    def query(self, text: str, n: int = 5) -> List[Memory]:
        if not self.prototypes:
            return []
        embed = np.array(self.embedder.embed(text))
        arr = np.stack(self.proto_embeds)
        proto_dists = np.linalg.norm(arr - embed, axis=1)
        proto_order = proto_dists.argsort()[:n]
        scored: list[tuple[float, Memory]] = []
        for idx in proto_order:
            pid = self.prototypes[idx]
            for mem, memb in zip(self.memories, self.mem_embeds):
                if mem.prototype_id != pid:
                    continue
                dist = float(np.linalg.norm(embed - memb))
                scored.append((dist, mem))
        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored[:n]]

    def decode_prototype(self, pid: str, n: int = 1) -> List[Memory]:
        if pid not in self.prototypes:
            return []
        idx = self.prototypes.index(pid)
        embed = self.proto_embeds[idx]
        scored: list[tuple[float, Memory]] = []
        for mem, memb in zip(self.memories, self.mem_embeds):
            if mem.prototype_id != pid:
                continue
            dist = float(np.linalg.norm(embed - memb))
            scored.append((dist, mem))
        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored[:n]]

    def summarize_prototype(self, pid: str, max_words: int = 50) -> str | None:
        mems = self.decode_prototype(pid, n=5)
        if not mems:
            return None
        text = " ".join(m.text for m in mems)
        words = text.split()
        return " ".join(words[:max_words])

    def dump_memories(self, prototype_id: str | None = None) -> List[Memory]:
        mems = [
            m
            for m in self.memories
            if prototype_id is None or m.prototype_id == prototype_id
        ]
        return list(mems)


class ChromaVectorStore(JSONVectorStore):
    """ChromaDB-backed store."""

    def __init__(
        self,
        client: Optional["chromadb.Client"] = None,
        threshold: float = 0.4,
        *,
        min_threshold: float = 0.05,
        decay_exponent: float = 0.5,
        embedder: Embedder | None = None,
    ) -> None:
        if chromadb is None:
            raise ImportError("chromadb is required for ChromaVectorStore")
        super().__init__(
            path=None,
            threshold=threshold,
            min_threshold=min_threshold,
            decay_exponent=decay_exponent,
            embedder=embedder,
        )
        self.client = client or default_chroma_client()
        self.proto_collection = self.client.get_or_create_collection("prototypes")
        self.memory_collection = self.client.get_or_create_collection("memories")

    # override persistence methods with chroma logic ---------------------
    def _save(self) -> None:  # pragma: no cover - not used
        pass

    def add_memory(self, text: str) -> Memory:
        embed = np.array(self.embedder.embed(text))
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
        self.memory_collection.add(
            ids=[mid],
            embeddings=[embed],
            metadatas=[{"prototype_id": pid}],
            documents=[text],
        )
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

    def decode_prototype(self, pid: str, n: int = 1) -> List[Memory]:
        proto = self.proto_collection.get(ids=[pid], include=["embeddings"])
        if not proto["ids"] or proto.get("embeddings") is None:
            return []
        embed = np.array(proto["embeddings"][0])
        mem_res = self.memory_collection.get(
            where={"prototype_id": pid}, include=["embeddings", "documents"]
        )
        if not mem_res["ids"]:
            return []
        scored: list[tuple[float, Memory]] = []
        for mid, doc, memb in zip(
            mem_res["ids"], mem_res["documents"], mem_res["embeddings"]
        ):
            dist = float(np.linalg.norm(embed - np.array(memb)))
            scored.append((dist, Memory(id=mid, text=doc, prototype_id=pid)))
        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored[:n]]

    def dump_memories(self, prototype_id: str | None = None) -> List[Memory]:
        where = {"prototype_id": prototype_id} if prototype_id else None
        mem_res = self.memory_collection.get(
            where=where, include=["metadatas", "documents"]
        )
        memories: list[Memory] = []
        for mid, doc, meta in zip(
            mem_res.get("ids", []),
            mem_res.get("documents", []),
            mem_res.get("metadatas", []),
        ):
            pid = meta.get("prototype_id") if meta else None
            memories.append(Memory(id=mid, text=doc, prototype_id=pid))
        return memories

    def _create_prototype(self, embed: np.ndarray) -> str:
        pid = str(uuid.uuid4())
        self.proto_collection.add(ids=[pid], embeddings=[embed])
        return pid

    def _update_prototype(
        self, pid: str, embed: np.ndarray, alpha: float = 0.1
    ) -> None:
        proto = self.proto_collection.get(ids=[pid], include=["embeddings"])
        if not proto["ids"]:
            self.proto_collection.add(ids=[pid], embeddings=[embed])
            return
        old = np.array(proto["embeddings"][0])
        new = (1 - alpha) * old + alpha * embed
        self.proto_collection.update(ids=[pid], embeddings=[new])

    def _adapt_threshold(self) -> None:
        count = self.proto_collection.count()
        if count <= 1:
            self.threshold = self.base_threshold
        else:
            adjusted = self.base_threshold / (count**self.decay_exponent)
            self.threshold = max(self.min_threshold, adjusted)

    def prototype_count(self) -> int:
        return self.proto_collection.count()


class CloudVectorStore(JSONVectorStore):
    """Placeholder for a remote service-backed store."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        *,
        path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(path=path, **kwargs)
        self.base_url = base_url
        self.api_key = api_key

    def add_memory(self, text: str) -> Memory:  # pragma: no cover - stub
        raise NotImplementedError("CloudVectorStore not implemented yet")

    def query(self, text: str, n: int = 5) -> List[Memory]:  # pragma: no cover - stub
        raise NotImplementedError("CloudVectorStore not implemented yet")

    def decode_prototype(
        self, pid: str, n: int = 1
    ) -> List[Memory]:  # pragma: no cover - stub
        raise NotImplementedError("CloudVectorStore not implemented yet")

    def dump_memories(
        self, prototype_id: str | None = None
    ) -> List[Memory]:  # pragma: no cover - stub
        raise NotImplementedError("CloudVectorStore not implemented yet")


def default_chroma_client() -> "chromadb.Client":  # pragma: no cover - helper
    if chromadb is None:
        raise ImportError("chromadb is required for ChromaVectorStore")
    path = os.path.join(os.getcwd(), "gist_memory_db")
    return chromadb.PersistentClient(path)


# Backwards compatible alias
PrototypeStore = JSONVectorStore
summarize_prototype = JSONVectorStore.summarize_prototype

__all__ = [
    "Memory",
    "Prototype",
    "VectorStore",
    "JSONVectorStore",
    "ChromaVectorStore",
    "CloudVectorStore",
    "PrototypeStore",
    "default_chroma_client",
    "summarize_prototype",
]
