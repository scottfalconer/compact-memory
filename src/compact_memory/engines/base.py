from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import json
import os
import uuid
import inspect
import time
import sys # For printing warnings
from pathlib import Path # Added import

import numpy as np
from pydantic import BaseModel # For isinstance check in load

from ..chunker import Chunker, SentenceWindowChunker, FixedSizeChunker
# Ensure EngineConfig is imported for type hinting and instantiation
from ..engine_config import EngineConfig
from ..embedding_pipeline import embed_text, get_embedding_dim # get_embedding_dim is imported here
from ..utils import calculate_sha256
from ..models import BeliefPrototype, RawMemory
from ..vector_store import (
    VectorStore,
    create_vector_store,
    InMemoryVectorStore # Added for isinstance check in load
)


@dataclass
class CompressedMemory:
    text: str
    metadata: Optional[Dict[str, Any]] = None
    engine_id: Optional[str] = None
    engine_config: Optional[Dict[str, Any]] = None
    trace: Optional[CompressionTrace] = None


@dataclass
class CompressionTrace:
    engine_name: str
    strategy_params: Dict[str, Any]
    input_summary: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    processing_ms: float | None = None
    final_compressed_object_preview: Optional[str] = None
    original_tokens: int | None = None
    compressed_tokens: int | None = None

    def add_step(self, step_type: str, details: Dict[str, Any]) -> None:
        self.steps.append({"type": step_type, "details": details})


class BaseCompressionEngine:
    id = "base"

    def __init__(
        self,
        *,
        chunker: Chunker | None = None,
        embedding_fn: Callable[[str | Sequence[str]], np.ndarray] = embed_text,
        preprocess_fn: Callable[[str], str] | None = None,
        vector_store: VectorStore | None = None,
        config: Optional[EngineConfig | Dict[str, Any]] = None,
        **kwargs,
    ) -> None:

        processed_config_data: Dict[str, Any] = {}
        if isinstance(config, EngineConfig):
            processed_config_data = config.model_dump(exclude_unset=False)
            if config.model_extra:
                processed_config_data.update(config.model_extra)
        elif isinstance(config, dict):
            processed_config_data = config.copy()

        processed_config_data.update(kwargs)

        if isinstance(chunker, Chunker):
            processed_config_data["chunker_id"] = getattr(chunker, 'id', type(chunker).__name__)
        elif chunker is not None:
            processed_config_data["chunker_id"] = str(chunker)

        if isinstance(vector_store, VectorStore):
            processed_config_data["vector_store"] = getattr(vector_store, 'id', type(vector_store).__name__)
            if hasattr(vector_store, 'embedding_dim') and vector_store.embedding_dim is not None:
                processed_config_data["embedding_dim"] = vector_store.embedding_dim
        elif vector_store is not None:
            processed_config_data["vector_store"] = str(vector_store)

        self.config: EngineConfig = EngineConfig(**processed_config_data)

        if self.config.embedding_dim is None:
            if embedding_fn is embed_text: # Check for default function identity
                self.config.embedding_dim = get_embedding_dim() # Call the imported get_embedding_dim

        if chunker:
            self._chunker = chunker
        else:
            chunker_id_to_create = self.config.chunker_id
            if chunker_id_to_create == "fixed_size":
                self._chunker = FixedSizeChunker()
            elif chunker_id_to_create == "sentence_window":
                self._chunker = SentenceWindowChunker()
            else:
                print(f"Warning: Unknown chunker_id '{chunker_id_to_create}' in config. Defaulting to SentenceWindowChunker.", file=sys.stderr)
                self._chunker = SentenceWindowChunker()

        self.embedding_fn = embedding_fn
        self.preprocess_fn = preprocess_fn
        sig = inspect.signature(self.embedding_fn)
        self._embed_accepts_preprocess = "preprocess_fn" in sig.parameters

        if vector_store:
            self.vector_store = vector_store
        else:
            if self.config.embedding_dim is None:
                raise ValueError(
                    "embedding_dim could not be resolved for vector store creation."
                )
            self.vector_store = create_vector_store(
                store_type=self.config.vector_store,
                embedding_dim=self.config.embedding_dim,
                path=self.config.vector_store_path
            )

        self.memories: List[Dict[str, Any]] = []
        self.memory_hashes: Set[str] = set()

    def _compress_chunk(self, chunk_text: str) -> str:
        return chunk_text

    def _ensure_index(self) -> None:
        pass

    def _embed(self, text_or_texts: str | Sequence[str]) -> np.ndarray:
        if self._embed_accepts_preprocess:
            return self.embedding_fn(text_or_texts, preprocess_fn=self.preprocess_fn) # type: ignore
        processed_input = text_or_texts
        if self.preprocess_fn is not None:
            if isinstance(text_or_texts, str):
                processed_input = self.preprocess_fn(text_or_texts)
            else:
                processed_input = [self.preprocess_fn(t) for t in text_or_texts]
        return self.embedding_fn(processed_input)

    def ingest(self, text: str) -> List[str]:
        raw_chunks = self.chunker.chunk(text)
        if not raw_chunks: return []
        processed_chunks = [self._compress_chunk(chunk) for chunk in raw_chunks]

        vecs = self._embed(processed_chunks)
        if vecs.ndim == 1: vecs = vecs.reshape(1, -1)

        ids: List[str] = []
        for processed_chunk_text, vec in zip(processed_chunks, vecs):
            chunk_hash = calculate_sha256(processed_chunk_text)
            if chunk_hash not in self.memory_hashes:
                mid = uuid.uuid4().hex
                self.memories.append({"id": mid, "text": processed_chunk_text})
                if hasattr(self.vector_store, 'add_prototype'):
                     proto = BeliefPrototype(prototype_id=mid, vector_row_index=0)
                     self.vector_store.add_prototype(proto, vec.astype(np.float32))
                     mem = RawMemory(memory_id=mid, raw_text_hash=chunk_hash, raw_text=processed_chunk_text, embedding=vec.tolist())
                     if hasattr(self.vector_store, 'add_memory'): # Some VS might not have this
                        self.vector_store.add_memory(mem) # type: ignore
                elif hasattr(self.vector_store, 'add_with_id'):
                     self.vector_store.add_with_id(mid, vec.astype(np.float32), processed_chunk_text) # type: ignore
                else:
                    # Fallback or error if no known add method
                    print("Warning: Vector store does not have a known method to add items (add_prototype or add_with_id).", file=sys.stderr)


                ids.append(mid)
                self.memory_hashes.add(chunk_hash)
        return ids

    def recall(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        vs_count = 0
        if hasattr(self.vector_store, 'prototypes'):
            vs_count = len(self.vector_store.prototypes) # type: ignore

        if not self.memories and vs_count == 0 : return []
        qvec = self._embed(query).reshape(1, -1).astype(np.float32)

        k = min(top_k, vs_count if vs_count > 0 else (len(self.memories) if self.memories else 0) )
        if k == 0: return []

        nearest_ids_scores = self.vector_store.find_nearest(qvec[0], k)

        mem_lookup = {m["id"]: m["text"] for m in self.memories}
        results: List[Dict[str, Any]] = []
        for item_id, score in nearest_ids_scores:
            text = mem_lookup.get(item_id, "")
            results.append({"id": item_id, "text": text, "score": float(score)})
        return results

    def compress(
        self, text: str, budget: int, previous_compression_result: Optional[CompressedMemory] = None,
    ) -> CompressedMemory:
        start = time.monotonic()
        truncated = text[:budget]
        trace = CompressionTrace(
            engine_name=self.id, strategy_params={"budget": budget},
            input_summary={"original_length": len(text)},
            output_summary={"compressed_length": len(truncated)},
            final_compressed_object_preview=truncated[:50],
        )
        trace.processing_ms = (time.monotonic() - start) * 1000
        return CompressedMemory(
            text=truncated, trace=trace, engine_id=self.id,
            engine_config=self.config.model_dump(mode='json')
        )

    def save(self, path: os.PathLike | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        with open(p / "entries.json", "w", encoding="utf-8") as fh:
            json.dump(self.memories, fh)

        # Save actual embeddings from the vector store if possible
        embeddings_to_save = np.array([])
        if hasattr(self.vector_store, 'proto_vectors') and self.vector_store.proto_vectors is not None:
            embeddings_to_save = self.vector_store.proto_vectors # type: ignore
        elif self.memories: # Fallback: re-embed memories if VS doesn't expose vectors
            print("Warning: Re-embedding memories for saving as vector store does not directly expose vectors.", file=sys.stderr)
            embeddings_to_save = self._embed([mem['text'] for mem in self.memories])

        np.save(p / "embeddings.npy", embeddings_to_save)

        manifest = {
            "engine_id": self.id,
            "engine_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "config": self.config.model_dump(mode='json'),
            "metadata": {
                "num_memories": len(self.memories),
                "embedding_dim": self.config.embedding_dim,
            },
        }
        with open(p / "engine_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        self.vector_store.save(str(p / "vector_store_data"))

    def load(self, path: os.PathLike | str) -> None:
        p = Path(path)
        manifest_path = p / "engine_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Engine manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        loaded_manifest_config_dict = manifest.get("config", {})

        # Create a new EngineConfig instance from loaded, ensuring defaults for new fields
        # and then update with any existing self.config values that weren't in manifest (e.g. runtime extras)
        # This merge logic might need refinement based on desired precedence.
        # For now, manifest config takes precedence for saved fields.
        current_config_extras = self.config.model_extra if self.config and self.config.model_extra else {}
        merged_load_config = {**current_config_extras, **loaded_manifest_config_dict}
        self.config = EngineConfig(**merged_load_config)


        metadata = manifest.get("metadata", {})
        metadata_embedding_dim = metadata.get("embedding_dim")
        authoritative_embedding_dim = metadata_embedding_dim

        if authoritative_embedding_dim is None: # Try from current config if not in metadata
            authoritative_embedding_dim = self.config.embedding_dim

        embeddings_array = np.load(p / "embeddings.npy")
        loaded_embeddings_dim = embeddings_array.shape[1] if embeddings_array.size > 0 else None

        if authoritative_embedding_dim is None and loaded_embeddings_dim is not None:
            authoritative_embedding_dim = loaded_embeddings_dim
            print(f"Warning: embedding_dim not in manifest or config, inferred from embeddings.npy ({authoritative_embedding_dim})", file=sys.stderr)

        if authoritative_embedding_dim is None:
             raise ValueError("Cannot determine embedding dimension for loading vector store.")

        if self.config.embedding_dim != authoritative_embedding_dim:
            print(f"Warning: EngineConfig embedding_dim ({self.config.embedding_dim}) "
                  f"mismatches effective dimension ({authoritative_embedding_dim}) from manifest/embeddings. "
                  "Using effective dimension.", file=sys.stderr)
            self.config.embedding_dim = authoritative_embedding_dim

        self.vector_store = create_vector_store(
            store_type=self.config.vector_store,
            embedding_dim=self.config.embedding_dim,
            path=self.config.vector_store_path
        )
        self.vector_store.load(str(p / "vector_store_data"))

        with open(p / "entries.json", "r", encoding="utf-8") as fh:
            self.memories = json.load(fh)

        self.memory_hashes = {calculate_sha256(mem["text"]) for mem in self.memories}

        if isinstance(self.vector_store, InMemoryVectorStore):
            self.vector_store.prototypes = []
            dim_for_empty_proto = self.config.embedding_dim if self.config.embedding_dim is not None else 0
            self.vector_store.proto_vectors = np.zeros((0, dim_for_empty_proto), dtype=np.float32)
            self.vector_store.index = {}
            self.vector_store.memories = []

            if embeddings_array.size > 0 and len(self.memories) == embeddings_array.shape[0]:
                for i, mem_entry in enumerate(self.memories):
                    vec = embeddings_array[i]
                    proto = BeliefPrototype(prototype_id=mem_entry["id"], vector_row_index=i)
                    self.vector_store.add_prototype(proto, vec)
                    raw_mem = RawMemory(
                        memory_id=mem_entry["id"],
                        raw_text_hash=calculate_sha256(mem_entry["text"]),
                        raw_text=mem_entry["text"],
                        embedding=vec.tolist()
                    )
                    self.vector_store.add_memory(raw_mem)
                self.vector_store._index_dirty = True
            elif embeddings_array.size > 0:
                 print(f"Warning: Mismatch between loaded memories ({len(self.memories)}) "
                       f"and embeddings ({embeddings_array.shape[0]}). InMemoryVectorStore not fully populated.", file=sys.stderr)

        vs_count_for_load = 0
        if hasattr(self.vector_store, 'prototypes'):
            vs_count_for_load = len(self.vector_store.prototypes) # type: ignore

        if embeddings_array.size > 0 and vs_count_for_load != embeddings_array.shape[0]:
             print(f"Warning: Final vector store count ({vs_count_for_load}) and "
                   f"number of loaded embeddings ({embeddings_array.shape[0]}) mismatch after population attempt.", file=sys.stderr)

    @property
    def embeddings(self) -> np.ndarray:
        if hasattr(self.vector_store, 'proto_vectors') and self.vector_store.proto_vectors is not None:
            return self.vector_store.proto_vectors # type: ignore
        dim = self.config.embedding_dim if self.config.embedding_dim is not None else 0
        if dim == 0 :
            dim = get_embedding_dim()
        return np.zeros((0, dim), dtype=np.float32)

    @property
    def chunker(self) -> Chunker:
        return self._chunker

    @chunker.setter
    def chunker(self, value: Chunker) -> None:
        if not isinstance(value, Chunker):
            raise TypeError("chunker must implement Chunker interface")
        self._chunker = value
        if isinstance(value, FixedSizeChunker):
            self.config.chunker_id = "fixed_size"
        elif isinstance(value, SentenceWindowChunker):
            self.config.chunker_id = "sentence_window"
        else:
            self.config.chunker_id = getattr(value, 'id', type(value).__name__)

__all__ = [
    "BaseCompressionEngine",
    "CompressedMemory",
    "CompressionTrace",
    "EngineConfig"
]
