from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from compact_memory.embedding_pipeline import embed_text, get_embedding_dim

from .episode import Episode


class EpisodeStorage:
    """Persist episodes and support vector search."""

    def __init__(self, path: Path, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model
        self.embedding_dim = get_embedding_dim(model_name=embedding_model)
        self.path.mkdir(parents=True, exist_ok=True)
        self.episodes: List[Episode] = []
        self.embeddings: np.ndarray = np.zeros(
            (0, self.embedding_dim), dtype=np.float32
        )
        self._episodes_file = self.path / "episodes.jsonl"
        self._vectors_file = self.path / "episode_vectors.npy"
        if self._episodes_file.exists():
            self._load()

    # ---------------------------------------------------------
    def _load(self) -> None:
        with open(self._episodes_file) as f:
            for line in f:
                if line.strip():
                    ep = Episode.from_dict(json.loads(line))
                    self.episodes.append(ep)
        if self._vectors_file.exists():
            self.embeddings = np.load(self._vectors_file)
        if len(self.embeddings) != len(self.episodes):
            self.embeddings = np.zeros(
                (len(self.episodes), self.embedding_dim), dtype=np.float32
            )

    # ---------------------------------------------------------
    def add_episode(self, episode: Episode) -> None:
        vec = embed_text(episode.summary_gist, model_name=self.embedding_model)
        vec = vec.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        vec /= np.linalg.norm(vec) or 1.0
        self.episodes.append(episode)
        self.embeddings = np.vstack([self.embeddings, vec])
        with open(self._episodes_file, "a") as f:
            f.write(json.dumps(episode.to_dict()) + "\n")
        np.save(self._vectors_file, self.embeddings)

    # ---------------------------------------------------------
    def update_episode(self, episode: Episode) -> None:
        for idx, ep in enumerate(self.episodes):
            if ep.id == episode.id:
                self.episodes[idx] = episode
                break
        else:
            self.add_episode(episode)
            return
        with open(self._episodes_file, "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep.to_dict()) + "\n")
        np.save(self._vectors_file, self.embeddings)

    # ---------------------------------------------------------
    def query(self, text: str, k: int = 3) -> List[Episode]:
        if not self.episodes:
            return []
        vec = embed_text(text, model_name=self.embedding_model)
        vec = vec.astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        sims = self.embeddings @ vec
        idxs = np.argsort(sims)[::-1][:k]
        return [self.episodes[i] for i in idxs]
