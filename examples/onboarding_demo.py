"""
Demonstrates key features of the Compact Memory platform, emphasizing its role
as an **Experimentation Platform** for compression strategies.

This script covers:
1.  **Defining a Custom Strategy:** Shows the basic structure of a `CompressionStrategy`
    (using a simple `TruncateStrategy` as an example).
2.  **Data Ingestion:** Uses `ExperimentConfig` and `run_experiment` to process data,
    which can be a preliminary step in an experimental pipeline.
3.  **Direct Compression:** Illustrates how to use a strategy programmatically for
    direct text compression and inspection of results.
4.  **Running Experiments:**
    *   Uses `HistoryExperimentConfig` and `run_history_experiment` to evaluate
        a strategy's impact on maintaining dialogue history coherence.
    *   Uses `ResponseExperimentConfig` and `run_response_experiment` to assess
        how a strategy affects the quality of LLM responses generated using
        compressed context.
    *   Highlights the use of `param_grid` for systematically testing different
        configurations and comparing their outcomes.

The overall goal is to show how developers can leverage Compact Memory to
create, test, and validate different memory compression techniques.
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


# This TruncateStrategy is a very basic example. Its main purpose in this demo
# is to illustrate how a custom strategy is defined, registered, and then used
# within the experiment runners. More sophisticated strategies would involve
# complex logic for selecting, summarizing, or transforming text.
# As an "Experimentation Platform", Compact Memory allows you to plug in such
# strategies and rigorously evaluate their performance using the experiment runners.
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
    print("Starting Compact Memory Onboarding Demo...")
    print("This demo showcases how the Compact Memory framework can be used as an experimentation platform.")
    print("You'll see how to define a strategy, ingest data, perform direct compression,")
    print("and run history and response experiments to evaluate strategy performance.")
    print("-" * 70)

    # --- Section 1: Basic Data Ingestion with ExperimentConfig ---
    print("\nSection 1: Basic Data Ingestion")
    print("Demonstrates using ExperimentConfig. While shown here with a default behavior,")
    print("you could specify different strategies in a config to see their effect on ingestion.")
    print("This is the first step in many experimentation pipelines: preparing and understanding your data.")
    data_file = (
        Path(__file__).parents[1] / "sample_data" / "moon_landing" / "01_landing.txt"
    )
    # ExperimentConfig can be used to run baseline processing or to apply a specific strategy during ingestion.
    # For experimentation, you might create multiple ExperimentConfig objects with different strategy settings
    # to compare their effects on how data is initially processed or on the resulting memory store.
    cfg = ExperimentConfig(dataset=data_file)
    metrics = run_experiment(cfg)
    print("\nIngestion metrics (using default experiment behavior):")
    print(yaml.safe_dump(metrics, sort_keys=False))
    print("--- End Section 1 ---")
    print("-" * 70)

    # --- Section 2: Direct Compression ---
    print("\nSection 2: Direct Compression with a Custom Strategy")
    print("Shows how a developer can programmatically invoke any registered compression strategy.")
    print("This is useful for quick tests, debugging, or integrating compression into custom workflows.")
    sample_text = Path(data_file).read_text()
    strategy = TruncateStrategy() # Initialize our custom strategy.
    print(f"Using strategy: {strategy.id}")
    # Get the trace by assigning the second element of the tuple to a variable
    compressed_memory, trace = strategy.compress(sample_text, llm_token_budget=40) # Compress to 40 chars
    print(f"Compressed preview: {compressed_memory.text!r}")
    print(f"Compression trace details: {trace.strategy_name}, input: {trace.input_summary.get('input_length')}, output: {trace.output_summary.get('final_length')}")
    print("--- End Section 2 ---")
    print("-" * 70)

    # --- Section 3: History and Response Experiments for Strategy Evaluation ---
    print("\nSection 3: Evaluating Strategies with History and Response Experiments")
    print("This is core to the 'Experimentation Platform' aspect.")
    print("We use HistoryExperimentConfig and ResponseExperimentConfig to systematically evaluate")
    print("how different strategies or configurations impact performance on specific tasks.")

    hist_dataset = Path(__file__).parents[1] / "tests" / "data" / "history_dialogues.yaml"
    resp_dataset = Path(__file__).parents[1] / "tests" / "data" / "response_dialogues.yaml"

    print(f"\nUsing registered strategy '{TruncateStrategy.id}' for these experiments.")
    print("The 'param_grid' allows testing different strategy parameters or system settings.")
    # The param_grid allows you to define variations in parameters.
    # For a real strategy, you might test different budget sizes, summarization techniques, etc.
    # The experiment runners will execute runs for each combination in the grid.
    params = [
        {"config_prompt_num_forced_recent_turns": 1, "_experiment_name": "Force 1 Recent Turn"},
        {"config_prompt_num_forced_recent_turns": 2, "_experiment_name": "Force 2 Recent Turns"},
    ]
    # Example: If TruncateStrategy took a 'mode' parameter, you could have:
    # params = [
    #     {"strategy_params": {"mode": "leading"}, "config_prompt_num_forced_recent_turns": 1},
    #     {"strategy_params": {"mode": "trailing"}, "config_prompt_num_forced_recent_turns": 1},
    # ]
    # The results will show metrics for each parameter combination, allowing comparison.

    hist_cfg = HistoryExperimentConfig(dataset=hist_dataset, param_grid=params)
    resp_cfg = ResponseExperimentConfig(dataset=resp_dataset, param_grid=params)

    print("\nRunning History Experiment (evaluates recall and context tracking)...")
    hist_results = run_history_experiment(hist_cfg)
    print("\nHistory experiment results:")
    print(yaml.safe_dump(hist_results, sort_keys=False))
    print("These results help quantify how well the memory (as shaped by the strategy) supports coherent dialogue.")

    print("\nRunning Response Experiment (evaluates quality of LLM responses with compressed context)...")
    resp_results = run_response_experiment(resp_cfg)
    print("\nResponse experiment results:")
    print(yaml.safe_dump(resp_results, sort_keys=False))
    print("These results help quantify the impact of compression on the final LLM output quality.")
    print("By comparing results from different strategies/params, developers can choose the best approach.")
    print("--- End Section 3 ---")
    print("-" * 70)
    print("Onboarding Demo Finished.")


if __name__ == "__main__":
    main()
