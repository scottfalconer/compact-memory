from __future__ import annotations

import logging
from typing import List, Optional

from .episode import Decision, Episode
from .storage import EpisodeStorage

logger = logging.getLogger(__name__)


class RationaleEpisodeManager:
    """Capture decisions with rationales and manage episodes."""

    def __init__(
        self, store: EpisodeStorage, *, importance_threshold: float = 0.7
    ) -> None:
        self.store = store
        self.importance_threshold = importance_threshold
        self._current: List[Decision] = []

    # -----------------------------------------------------
    def store_step(
        self,
        step_id: str,
        step_summary: str,
        rationale: str,
        importance: float = 0.0,
    ) -> None:
        logger.debug("Storing step %s rationale: %s", step_id, rationale)
        self._current.append(
            Decision(
                step_id=step_id,
                step_summary=step_summary,
                rationale=rationale,
                importance=float(importance),
            )
        )

    # -----------------------------------------------------
    def finalize_episode(
        self, summary_gist: str, tags: Optional[List[str]] = None
    ) -> Episode:
        episode = Episode(
            summary_gist=summary_gist, tags=tags or [], decisions=self._current
        )
        self.store.add_episode(episode)
        logger.info("Episode %s saved", episode.id)
        self._current = []
        return episode

    # -----------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> List[Episode]:
        episodes = self.store.query(query, k)
        if not episodes:
            logger.warning("No episodes retrieved for query: %s", query)
        return episodes
