"""Simple onboarding demo for Gist Memory."""
from pathlib import Path
import os

from gist_memory.store import PrototypeStore
from gist_memory.memory_creation import IdentityMemoryCreator
from gist_memory.embedder import get_embedder
from gist_memory.config import DEFAULT_BRAIN_PATH

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    folder = Path(__file__).parent / "moon_landing"
    texts = [p.read_text() for p in sorted(folder.glob("*.txt"))]

    store = PrototypeStore(
        path=DEFAULT_BRAIN_PATH,
        embedder=get_embedder("local", "all-MiniLM-L6-v2"),
    )
    creator = IdentityMemoryCreator()

    for text in texts:
        before = store.prototype_count()
        mem = store.add_memory(creator.create(text))
        after = store.prototype_count()
        action = "Created" if after > before else "Updated"
        print(f"{action} prototype {mem.prototype_id} with memory {mem.id}")

    print()
    print(f"Total memories: {len(store.memories)}")
    print(f"Total prototypes: {store.prototype_count()}")


if __name__ == "__main__":
    main()
