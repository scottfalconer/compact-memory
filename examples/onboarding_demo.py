"""Simple onboarding demo for Gist Memory."""
from pathlib import Path

from gist_memory.store import PrototypeStore
from gist_memory.memory_creation import IdentityMemoryCreator
from gist_memory.embedder import get_embedder


def main() -> None:
    folder = Path(__file__).parent / "moon_landing"
    texts = [p.read_text() for p in sorted(folder.glob("*.txt"))]

    store = PrototypeStore(embedder=get_embedder("local", "all-MiniLM-L6-v2"))
    creator = IdentityMemoryCreator()

    for text in texts:
        before = store.proto_collection.count()
        mem = store.add_memory(creator.create(text))
        after = store.proto_collection.count()
        action = "Created" if after > before else "Updated"
        print(f"{action} prototype {mem.prototype_id} with memory {mem.id}")

    print()
    print(f"Total memories: {store.memory_collection.count()}")
    print(f"Total prototypes: {store.proto_collection.count()}")


if __name__ == "__main__":
    main()
