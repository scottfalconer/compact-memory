from __future__ import annotations

"""Helpers for running hyperparameter optimisation trials."""

from dataclasses import replace
from typing import Any, Callable, Dict

from .response_experiment import ResponseExperimentConfig, run_response_experiment
from .compression.strategies_abc import CompressionStrategy


def run_params_trial(
    params: Dict[str, Any] | Any,
    base_config: ResponseExperimentConfig,
    *,
    strategy_factory: Callable[[Dict[str, Any]], CompressionStrategy] | None = None,
    metric_id: str = "exact_match",
) -> float:
    """Run a single experiment with ``params`` and return ``metric_id`` score.

    ``params`` may be a plain dictionary or an object with a ``params``
    attribute (e.g. :class:`optuna.trial.Trial`). If ``strategy_factory`` is
    provided it will be called with ``params`` to instantiate the compression
    strategy used during the experiment.
    """

    if not isinstance(params, dict):
        if hasattr(params, "params"):
            params = dict(params.params)
        else:
            raise TypeError("params must be dict-like or have a 'params' attribute")

    cfg = replace(base_config, param_grid=[params])
    strategy = strategy_factory(params) if strategy_factory else None
    results = run_response_experiment(cfg, strategy)
    if not results:
        return 0.0
    metric = results[0].get("metrics", {}).get(metric_id, {})
    if not metric:
        return 0.0
    return float(next(iter(metric.values())))


__all__ = ["run_params_trial"]
