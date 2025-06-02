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
from .embedding_pipeline import embed_text, get_embedding_dim
from .chunker import SentenceWindowChunker
from .registry import get_validation_metric_class
from . import validation  # ensure metrics are registered
from .compression.strategies_abc import CompressionStrategy


@dataclass
class ResponseExperimentConfig:
    """Configuration for :func:`run_response_experiment`."""

    dataset: Path
    param_grid: List[Dict[str, Any]]
    validation_metrics: List[Dict[str, Any]] | None = None


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


def _evaluate_sample(
    sample: Dict[str, Any],
    params: Dict[str, Any],
    metric_entries: List[Dict[str, Any]],
    strategy: CompressionStrategy | None,
) -> Dict[str, Any]:
    mgr = ActiveMemoryManager(**params)

    work_dir = tempfile.mkdtemp()
    dim = get_embedding_dim()
    store = JsonNpyVectorStore(path=work_dir, embedding_model="experiment", embedding_dim=dim)
    agent = Agent(store, chunker=SentenceWindowChunker())

    for turn in sample["turns"]:
        if isinstance(turn, dict):
            text = turn.get("text") or next(iter(turn.values()))
        else:
            text = str(turn)
        agent.add_memory(text)
        emb = embed_text(text)
        mgr.add_turn(ConversationTurn(text=text, turn_embedding=emb.tolist()))

    query = sample["query"]
    answer = sample.get("answer", "")
    if strategy is not None:
        reply, info = agent.process_conversational_turn(query, mgr, compression=strategy)
    else:
        reply, info = agent.process_conversational_turn(query, mgr)
    tokens = info.get("prompt_tokens", 0)
    compression_ms = 0.0
    trace = info.get("compression_trace")
    if trace and isinstance(trace, dict):
        compression_ms = trace.get("processing_ms", 0.0) or 0.0

    metric_scores: Dict[str, Dict[str, float]] = {}
    for entry in metric_entries:
        MetricClass = get_validation_metric_class(entry["id"])
        metric = MetricClass(**entry.get("params", {}))
        metric_scores[entry["id"]] = metric.evaluate(
            llm_response=reply,
            reference_answer=answer,
            original_query=query,
        )

    return {"metrics": metric_scores, "prompt_tokens": tokens, "compression_ms": compression_ms}


def run_response_experiment(
    config: ResponseExperimentConfig,
    strategy: CompressionStrategy | None = None,
) -> List[Dict[str, Any]]:
    """Run the experiment defined by ``config``."""

    dataset = _load_dataset(config.dataset)
    metric_entries = config.validation_metrics or [{"id": "exact_match", "params": {}}]

    results = []
    for params in config.param_grid:
        aggregates: Dict[str, Dict[str, float]] = {
            m["id"]: {} for m in metric_entries
        }
        total_tokens = 0
        total_time = 0.0
        for sample in dataset:
            res = _evaluate_sample(sample, params, metric_entries, strategy)
            total_tokens += res["prompt_tokens"]
            total_time += res.get("compression_ms", 0.0)
            for mid, scores in res["metrics"].items():
                agg = aggregates[mid]
                for name, val in scores.items():
                    agg[name] = agg.get(name, 0.0) + val
        n = len(dataset) or 1
        avg_scores = {
            mid: {name: val / n for name, val in scores.items()}
            for mid, scores in aggregates.items()
        }
        results.append(
            {
                "params": params,
                "avg_prompt_tokens": total_tokens / n,
                "avg_compression_ms": total_time / n,
                "metrics": avg_scores,
            }
        )
    return results


__all__ = ["ResponseExperimentConfig", "run_response_experiment"]

