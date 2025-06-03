"""
Demonstrates key features of the Compact Memory platform, including custom
strategy definition, data ingestion, direct compression, and running
history/response experiments.
"""
from __future__ import annotations

from pathlib import Path
import yaml

from compact_memory import (
    ExperimentConfig,
    HistoryExperimentConfig,
    ResponseExperimentConfig,
    run_experiment,
    run_history_experiment,
    run_response_experiment,
)
from compact_memory.compression import (
    CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)
from compact_memory.registry import register_compression_strategy


class TruncateStrategy(CompressionStrategy):
    """
    Minimalistic example `CompressionStrategy` for demonstration purposes.
    This shows the basic structure required to implement the CompressionStrategy interface.
    """

    # The unique identifier for this strategy, used for registration and selection.
    id = "truncate"

    def compress(
        self, text_or_chunks: str | list[str], llm_token_budget: int, **kwargs
    ) -> tuple[CompressedMemory, CompressionTrace]:
        """
        Compresses the input text by simple truncation.

        Args:
            text_or_chunks: The input text or list of text chunks to compress.
            llm_token_budget: The target maximum number of characters (acting as a proxy for tokens here).
            **kwargs: Additional keyword arguments (not used in this simple strategy).

        Returns:
            A tuple containing:
                - CompressedMemory: An object holding the compressed text.
                - CompressionTrace: An object detailing the steps and outcomes of the compression.
        """
        if isinstance(text_or_chunks, list):
            text = " ".join(text_or_chunks)
        else:
            text = str(text_or_chunks)
        # CompressedMemory stores the result of the compression.
        compressed = CompressedMemory(text=text[:llm_token_budget])
        # CompressionTrace records metadata about the compression process, useful for debugging and analysis.
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
    # --- Section 1: Basic Data Ingestion ---
    # This section demonstrates basic data ingestion using ExperimentConfig and run_experiment.
    # ExperimentConfig, when run without a specific strategy, might perform baseline processing
    # or use a default strategy if one is defined. Here, it primarily shows how data is loaded.
    data_file = (
        Path(__file__).parents[1] / "sample_data" / "moon_landing" / "01_landing.txt"
    )
    cfg = ExperimentConfig(dataset=data_file)  # No strategy specified, uses default behavior.
    metrics = run_experiment(cfg)
    print("Ingestion metrics:\n" + yaml.safe_dump(metrics, sort_keys=False))
    print("\n--- This shows baseline metrics from processing the data file. --- \n")

    # --- Section 2: Direct Compression ---
    # Demonstrate direct use of a compression strategy.
    sample_text = Path(data_file).read_text()
    # Initialize our custom strategy.
    strategy = TruncateStrategy()
    # Compress the sample text to a specific token budget.
    compressed, _ = strategy.compress(sample_text, llm_token_budget=40)
    print(f"Compressed preview: {compressed.text!r}\n")
    print("\n--- The above shows the text directly compressed by TruncateStrategy. --- \n")

    # --- Section 3: History and Response Experiments ---
    # Evaluate retrieval and response quality using example dialogue datasets.
    # HistoryExperimentConfig and ResponseExperimentConfig are used to set up these specific types of experiments.
    # The `param_grid` allows testing different configurations of the strategy or memory system.
    hist_dataset = Path(__file__).parents[1] / "tests" / "data" / "history_dialogues.yaml"
    resp_dataset = Path(__file__).parents[1] / "tests" / "data" / "response_dialogues.yaml"
    params = [
        # Example: test with different numbers of recent turns forced into context
        {"config_prompt_num_forced_recent_turns": 1},
        {"config_prompt_num_forced_recent_turns": 2},
    ]
    # Note: These experiments will use the registered 'truncate' strategy if not overridden by
    # a default in the experiment runner or if the configs were to specify a strategy.
    # For this demo, the effect of TruncateStrategy on dialogue tasks might be limited,
    # but it demonstrates the workflow.
    hist_cfg = HistoryExperimentConfig(dataset=hist_dataset, param_grid=params)
    resp_cfg = ResponseExperimentConfig(dataset=resp_dataset, param_grid=params)

    hist_results = run_history_experiment(hist_cfg)
    resp_results = run_response_experiment(resp_cfg)

    print("History experiment results:\n" + yaml.safe_dump(hist_results, sort_keys=False))
    print("\n--- History experiment evaluates memory's ability to support coherent multi-turn dialogue (e.g., recall quality). --- \n")
    print("Response experiment results:\n" + yaml.safe_dump(resp_results, sort_keys=False))
    print("\n--- Response experiment evaluates the quality of LLM responses generated using the compressed memory. --- \n")


if __name__ == "__main__":
    main()
