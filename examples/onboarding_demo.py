"""Simple onboarding demo for Gist Memory."""
from pathlib import Path
import os

from gist_memory import Agent, JsonNpyVectorStore
from gist_memory.embedding_pipeline import embed_text, get_embedding_dim
from gist_memory.config import DEFAULT_BRAIN_PATH

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    folder = Path(__file__).parent / "moon_landing"
    texts = [p.read_text() for p in sorted(folder.glob("*.txt"))]

    path = Path(DEFAULT_BRAIN_PATH)
    if (path / "meta.yaml").exists():
        store = JsonNpyVectorStore(str(path))
    else:
        dim = get_embedding_dim()
        store = JsonNpyVectorStore(
            str(path), embedding_model="all-MiniLM-L6-v2", embedding_dim=dim
        )
    agent = Agent(store)

    for text in texts:
        results = agent.add_memory(text)
        for res in results:
            action = "Created" if res.get("spawned") else "Updated"
            pid = res.get("prototype_id")
            print(f"{action} prototype {pid}")

    print()
    print(f"Total memories: {len(agent.store.memories)}")
    print(f"Total prototypes: {len(agent.store.prototypes)}")


if __name__ == "__main__":
    main()
