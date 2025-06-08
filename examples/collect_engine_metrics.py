from __future__ import annotations

"""Utility to collect basic metrics for available compression engines."""

import json
from pathlib import Path
from typing import Callable

from compact_memory.contrib import enable_all_experimental_engines
from compact_memory.engines.registry import (
    available_engines,
    get_compression_engine,
    register_compression_engine,
)
from compact_memory.validation.registry import get_validation_metric_class
import compact_memory.embedding_pipeline as ep
from compact_memory.embedding_pipeline import MockEncoder


def main(output_file: str = "engine_metrics.json") -> None:
    """Run each engine on sample text and record metrics."""
    # Use deterministic mock embeddings to avoid heavy downloads

    enable_all_experimental_engines()

    from compact_memory.engines import pipeline_engine

    register_compression_engine(
        pipeline_engine.PipelineEngine.id, pipeline_engine.PipelineEngine
    )

    engines = list(available_engines())

    text = Path("sample_data/moon_landing/full.txt").read_text()

    ratio_metric = get_validation_metric_class("compression_ratio")()
    embed_metric = get_validation_metric_class("embedding_similarity")()

    results: dict[str, dict[str, float]] = {}

    for eng_id in engines:
        EngineCls = get_compression_engine(eng_id)
        if eng_id == "pipeline":
            from compact_memory.engines.first_last_engine import FirstLastEngine
            from compact_memory.engines.no_compression_engine import NoCompressionEngine

            engine = EngineCls([FirstLastEngine(), NoCompressionEngine()])
        else:
            engine = EngineCls()

        if eng_id == "none":
            # Disable truncation for the no-op engine so the metrics reflect
            # a true no-compression baseline.
            compressed, _ = engine.compress(text, llm_token_budget=None)
        else:
            result = engine.compress(text, llm_token_budget=100)
            if isinstance(result, tuple):
                compressed, _ = result
            else:
                compressed = result

        if hasattr(compressed, "text"):
            comp_text = compressed.text
        else:
            comp_text = compressed.get("content", str(compressed))
        ratio = ratio_metric.evaluate(original_text=text, compressed_text=comp_text)[
            "compression_ratio"
        ]
        embed_score = embed_metric.evaluate(
            original_text=text, compressed_text=comp_text
        )["semantic_similarity"]
        results[eng_id] = {
            "compression_ratio": ratio,
            "embedding_similarity": embed_score,
        }

    Path(output_file).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
