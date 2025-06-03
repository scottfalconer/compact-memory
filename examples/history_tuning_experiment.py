"""Example script to tune ActiveMemoryManager history parameters."""
from pathlib import Path
import yaml

from compact_memory.history_experiment import HistoryExperimentConfig, run_history_experiment


def main() -> None:
    # Use the shared test dataset packaged under tests/data
    dataset = Path(__file__).parents[1] / "tests" / "data" / "history_dialogues.yaml"
    param_grid = [
        {
            "config_prompt_num_forced_recent_turns": 1,
            "config_prompt_max_activated_older_turns": 1,
            "config_relevance_boost_factor": 1.0,
        },
        {
            "config_prompt_num_forced_recent_turns": 2,
            "config_prompt_max_activated_older_turns": 2,
            "config_relevance_boost_factor": 1.5,
        },
    ]
    cfg = HistoryExperimentConfig(dataset=dataset, param_grid=param_grid)
    results = run_history_experiment(cfg)
    print(yaml.safe_dump(results, sort_keys=False))


if __name__ == "__main__":
    main()
