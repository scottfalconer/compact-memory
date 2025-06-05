from __future__ import annotations

"""Demonstrate prototype evolution when related memories are added."""

from pathlib import Path
from tempfile import TemporaryDirectory

from compact_memory.vector_store import InMemoryVectorStore
from CompressionEngine.contrib.prototype_engine import PrototypeEngine # Updated import
from compact_memory.embedding_pipeline import MockEncoder
from compact_memory import embedding_pipeline as agent_mod # agent_mod might be misleading later


def main() -> None:
    enc = MockEncoder()

    def embed(texts):
        return enc.encode(texts)

    agent_mod.embed_text = embed  # type: ignore

    with TemporaryDirectory() as _:
        store = InMemoryVectorStore(embedding_dim=enc.dim)
        engine = PrototypeEngine(store) # Updated class name and variable
        texts = [
            "Neil Armstrong was the first person to walk on the moon.",
            "Armstrong's lunar landing occurred in 1969.",
            "The Apollo 11 mission carried Neil Armstrong to the moon.",
        ]
        for t in texts:
            engine.add_memory(t) # Updated variable
            proto = engine.store.prototypes[0] # Updated variable
            print(f"After ingesting: {t}")
            print(f"  Summary: {proto.summary_text}")
            print(f"  Strength: {proto.strength}\n")


if __name__ == "__main__":
    main()
