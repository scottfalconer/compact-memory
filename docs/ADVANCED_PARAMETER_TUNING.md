# Advanced Parameter Tuning

Gist Memory experiments can be optimised with external Hyperparameter
Optimisation (HPO) tools such as **Optuna** or **Ray Tune**. The
`gist_memory.hpo.run_params_trial` helper executes a single experiment
with a set of parameters and returns the chosen validation metric. This
makes it easy to plug Gist Memory into your favourite HPO library.

## Quick Example using Optuna

```python
from pathlib import Path
import optuna
from gist_memory.response_experiment import ResponseExperimentConfig
from gist_memory.hpo import run_params_trial

base = ResponseExperimentConfig(
    dataset=Path("dialogues.yaml"),
    param_grid=[{}],  # replaced during optimisation
    validation_metrics=[{"id": "exact_match", "params": {}}],
)

 def objective(trial: optuna.Trial) -> float:
     params = {
         "config_prompt_num_forced_recent_turns": trial.suggest_int("turns", 1, 3),
         "config_relevance_boost_factor": trial.suggest_float("boost", 1.0, 2.0),
     }
     return run_params_trial(params, base)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print(study.best_params, study.best_value)
```

## CLI Integration

Create a Python script similar to the example above and run it with:

```bash
gist-memory experiment optimize path/to/optimize_script.py
```

The script has full access to the Gist Memory API, so you can define
custom search spaces or strategies.

## Optional Dependencies

Install `optuna` or `ray[tune]` to enable these features:

```bash
pip install gist-memory[optuna]
# or
pip install gist-memory[ray]
```
