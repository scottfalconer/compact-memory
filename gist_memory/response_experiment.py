from __future__ import annotations

"""Run end-to-end experiments evaluating prompt strategies."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import tempfile
import yaml

from .agent import Agent
from .active_memory_manager import ActiveMemoryManager, ConversationTurn
from .json_npy_store import JsonNpyVectorStore
from .embedding_pipeline import MockEncoder
from .chunker import SentenceWindowChunker
from .token_utils import token_count


@dataclass
class ResponseExperimentConfig:
    """Configuration for :func:`run_response_experiment`."""

    dataset: Path
    param_grid: List[Dict[str, Any]]


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    return list(data)


def _simple_f1(reference: str, prediction: str) -> float:
    ref = reference.split()
    pred = prediction.split()
    if not ref or not pred:
        return 0.0
    common = set(ref) & set(pred)
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _evaluate_sample(sample: Dict[str, Any], params: Dict[str, Any], enc: MockEncoder) -> Dict[str, Any]:
    mgr = ActiveMemoryManager(**params)

    work_dir = tempfile.mkdtemp()
    store = JsonNpyVectorStore(path=work_dir, embedding_model="experiment", embedding_dim=enc.dim)
    agent = Agent(store, chunker=SentenceWindowChunker())

    for turn in sample["turns"]:
        if isinstance(turn, dict):
            text = turn.get("text") or next(iter(turn.values()))
        else:
            text = str(turn)
        agent.add_memory(text)
        emb = enc.encode(text)
        mgr.add_turn(ConversationTurn(text=text, turn_embedding=emb.tolist()))

    query = sample["query"]
    answer = sample.get("answer", "")
    reply, info = agent.process_conversational_turn(query, mgr)
    score = _simple_f1(answer, reply)
    tokens = info.get("prompt_tokens", 0)
    return {"score": score, "prompt_tokens": tokens}


def run_response_experiment(config: ResponseExperimentConfig) -> List[Dict[str, Any]]:
    """Run the experiment defined by ``config``."""

    dataset = _load_dataset(config.dataset)
    enc = MockEncoder()
    results = []
    for params in config.param_grid:
        total_score = 0.0
        total_tokens = 0
        for sample in dataset:
            res = _evaluate_sample(sample, params, enc)
            total_score += res["score"]
            total_tokens += res["prompt_tokens"]
        n = len(dataset) or 1
        results.append(
            {
                "params": params,
                "avg_f1": total_score / n,
                "avg_prompt_tokens": total_tokens / n,
            }
        )
    return results


__all__ = ["ResponseExperimentConfig", "run_response_experiment"]

