from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml

from .active_memory_manager import ActiveMemoryManager, ConversationTurn
from .embedding_pipeline import MockEncoder
from .compression.strategies_abc import CompressionStrategy


@dataclass
class HistoryExperimentConfig:
    """Configuration for :func:`run_history_experiment`."""

    dataset: Path
    param_grid: List[Dict[str, Any]]


# --------------------------------------------------------------
def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    return list(data)


# --------------------------------------------------------------
def _evaluate_dialogue(sample: Dict[str, Any], params: Dict[str, Any], encoder: MockEncoder) -> bool:
    mgr = ActiveMemoryManager(**params)
    for turn in sample["turns"]:
        if isinstance(turn, dict):
            text = turn.get("text") or next(iter(turn.values()))
        else:
            text = str(turn)
        emb = encoder.encode(text)
        mgr.add_turn(ConversationTurn(text=text, turn_embedding=emb.tolist()))
    query = sample["query"]
    query_emb = encoder.encode(query)
    candidates = mgr.select_history_candidates_for_prompt(query_emb)
    idx = int(sample["required_turn_index"])
    req_turn = sample["turns"][idx]
    if isinstance(req_turn, dict):
        required_text = req_turn.get("text") or next(iter(req_turn.values()))
    else:
        required_text = str(req_turn)
    return any(t.text == required_text for t in candidates)


# --------------------------------------------------------------
def run_history_experiment(
    config: HistoryExperimentConfig,
    strategy: CompressionStrategy | None = None,
) -> List[Dict[str, Any]]:
    """Run history parameter tuning experiment on ``config.dataset``."""

    dataset = _load_dataset(config.dataset)
    encoder = MockEncoder()
    results = []
    for params in config.param_grid:
        successes = 0
        for sample in dataset:
            if _evaluate_dialogue(sample, params, encoder):
                successes += 1
        hit_rate = successes / len(dataset) if dataset else 0.0
        results.append({"params": params, "hit_rate": hit_rate})
    return results


__all__ = ["HistoryExperimentConfig", "run_history_experiment"]
