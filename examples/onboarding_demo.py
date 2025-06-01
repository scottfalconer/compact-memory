"""Quick start demonstration of the experimentation workflow."""
from __future__ import annotations

from pathlib import Path
import yaml

from gist_memory import (
    ExperimentConfig,
    HistoryExperimentConfig,
    ResponseExperimentConfig,
    run_experiment,
    run_history_experiment,
    run_response_experiment,
)
from gist_memory.compression import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)
from gist_memory.registry import register_compression_strategy


class TruncateStrategy(CompressionStrategy):
    """Very small example compression strategy."""

    id = "truncate"

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = str(text_or_chunks)
        compressed = CompressedMemory(text=text[:llm_token_budget])
        trace = CompressionTrace(
            strategy_name=self.id,
            strategy_params={"llm_token_budget": llm_token_budget},
            input_summary={"input_length": len(text)},
            steps=[{"type": "truncate", "new_length": len(compressed.text)}],
            output_summary={"final_length": len(compressed.text)},
            final_compressed_object_preview=compressed.text[:50],
        )
        return compressed, trace


register_compression_strategy(TruncateStrategy.id, TruncateStrategy)


def main() -> None:
    # Ingest a single moon landing excerpt
    data_file = (
        Path(__file__).parents[1] / "sample_data" / "moon_landing" / "01_landing.txt"
    )
    cfg = ExperimentConfig(dataset=data_file)
    metrics = run_experiment(cfg)
    print("Ingestion metrics:\n" + yaml.safe_dump(metrics, sort_keys=False))

    # Demonstrate compression on some text
    sample_text = Path(data_file).read_text()
    strategy = TruncateStrategy()
    compressed = strategy.compress(sample_text, llm_token_budget=40)
    print(f"Compressed preview: {compressed.text!r}\n")

    # Evaluate retrieval and response quality using small test datasets
    hist_dataset = Path(__file__).parents[1] / "tests" / "data" / "history_dialogues.yaml"
    resp_dataset = Path(__file__).parents[1] / "tests" / "data" / "response_dialogues.yaml"
    params = [
        {"config_prompt_num_forced_recent_turns": 1},
        {"config_prompt_num_forced_recent_turns": 2},
    ]
    hist_cfg = HistoryExperimentConfig(dataset=hist_dataset, param_grid=params)
    resp_cfg = ResponseExperimentConfig(dataset=resp_dataset, param_grid=params)
    hist_results = run_history_experiment(hist_cfg)
    resp_results = run_response_experiment(resp_cfg)
    print("History experiment results:\n" + yaml.safe_dump(hist_results, sort_keys=False))
    print("Response experiment results:\n" + yaml.safe_dump(resp_results, sort_keys=False))


if __name__ == "__main__":
    main()
