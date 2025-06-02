import json
import shutil
from pathlib import Path
from typing import Optional
import logging

import typer
import portalocker
from .token_utils import token_count
from rich.table import Table
from rich.console import Console

from .logging_utils import configure_logging


from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .embedding_pipeline import (
    get_embedding_dim,
    EmbeddingDimensionMismatchError,
)
from .utils import load_agent
from .config import DEFAULT_BRAIN_PATH
from .compression import available_strategies, get_compression_strategy
from .package_utils import (
    load_manifest,
    validate_manifest,
    load_strategy_class,
    validate_package_dir,
    check_requirements_installed,
)
from .response_experiment import ResponseExperimentConfig, run_response_experiment

app = typer.Typer(help="Gist Memory command line interface")
console = Console()
VERBOSE = False


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Write debug logs to this file"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Configure logging before executing commands."""
    if log_file:
        level = logging.DEBUG if verbose else logging.INFO
        configure_logging(log_file, level)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG)
    global VERBOSE
    VERBOSE = verbose
    ctx.obj = {"verbose": verbose}


class PersistenceLock:
    def __init__(self, path: Path) -> None:
        self.file = (path / ".lock").open("a+")

    def __enter__(self):
        portalocker.lock(self.file, portalocker.LockFlags.EXCLUSIVE)
        return self

    def __exit__(self, exc_type, exc, tb):
        portalocker.unlock(self.file)
        self.file.close()


def _corrupt_exit(path: Path, exc: Exception) -> None:
    typer.echo(f"Error: Brain data is corrupted. {exc}", err=True)
    typer.echo(
        f"Try running gist-memory validate {path} for more details or restore from a backup.",
        err=True,
    )
    raise typer.Exit(code=1)


def _load_agent(path: Path) -> Agent:
    try:
        return load_agent(path)
    except Exception as exc:
        _corrupt_exit(path, exc)


@app.command()
def init(
    directory: str = typer.Argument(DEFAULT_BRAIN_PATH),
    *,
    agent_name: str = "default",
    model_name: str = "all-MiniLM-L6-v2",
    tau: float = 0.8,
    alpha: float = 0.1,
    chunker: str = "sentence_window",
) -> None:
    """Create a new agent store."""
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        typer.secho(
            "Directory already exists and is not empty",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if not 0.5 <= tau <= 0.95:
        typer.secho(
            "Error: --tau must be between 0.5 and 0.95.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    try:
        dim = get_embedding_dim()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    store = JsonNpyVectorStore(
        path=str(path), embedding_model=model_name, embedding_dim=dim
    )
    store.meta.update(
        {
            "agent_name": agent_name,
            "tau": tau,
            "alpha": alpha,
            "chunker": chunker,
        }
    )
    store.save()
    typer.echo(f"Initialized agent at {directory}")


strategy_app = typer.Typer(help="Inspect compression strategies")
app.add_typer(strategy_app, name="strategy")

package_app = typer.Typer(help="Manage strategy packages")
app.add_typer(package_app, name="package")

experiment_app = typer.Typer(help="Run experiments")
app.add_typer(experiment_app, name="experiment")

trace_app = typer.Typer(help="Inspect compression traces")
app.add_typer(trace_app, name="trace")


@strategy_app.command("inspect")
def strategy_inspect(
    strategy_name: str,
    *,
    agent_name: str = typer.Option(DEFAULT_BRAIN_PATH, help="Agent directory"),
    list_prototypes: bool = typer.Option(
        False, "--list-prototypes", help="List prototypes for the strategy"
    ),
) -> None:
    """Inspect a specific LTM compression strategy."""

    if strategy_name != "prototype":
        typer.echo(f"Unknown strategy: {strategy_name}", err=True)
        raise typer.Exit(code=1)

    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    agent = _load_agent(path)

    if list_prototypes:
        protos = agent.get_prototypes_view()
        table = Table("id", "strength", "confidence", "summary", title="Beliefs")
        for p in protos:
            table.add_row(
                p["id"],
                f"{p['strength']:.2f}",
                f"{p['confidence']:.2f}",
                p["summary"][:60],
            )
        console.print(table)


@app.command()
def stats(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show statistics about the store."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    agent = _load_agent(path)
    data = agent.get_statistics()
    logging.debug("Collected statistics: %s", data)
    if json_output:
        typer.echo(json.dumps(data))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


@app.command()
def talk(
    *,
    agent_name: str = typer.Option(DEFAULT_BRAIN_PATH, help="Agent directory"),
    message: str = typer.Option(..., help="Message to send"),
    model_name: str = typer.Option("distilgpt2", help="Local chat model"),
    compression_strategy: Optional[str] = typer.Option(
        None,
        "--compression",
        help="Compression strategy",
        case_sensitive=False,
    ),
    show_prompt_tokens: bool = typer.Option(
        False,
        "--show-prompt-tokens",
        help="Display token count of the prompt sent to the LLM",
    ),
) -> None:
    """Talk to the brain using the same pathway as chat sessions."""

    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    with PersistenceLock(path):
        agent = _load_agent(path)

        from .local_llm import LocalChatModel

        try:
            agent._chat_model = LocalChatModel(model_name=model_name)
            agent._chat_model.load_model()
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)

        from .active_memory_manager import ActiveMemoryManager

        mgr = ActiveMemoryManager()
        comp = None
        if isinstance(compression_strategy, str) and compression_strategy:
            try:
                comp_cls = get_compression_strategy(compression_strategy)
                comp = comp_cls()
            except KeyError:
                typer.secho(
                    f"Unknown compression strategy '{compression_strategy}'",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

        try:
            result = agent.receive_channel_message(
                "cli", message, mgr, compression=comp
            )
        except TypeError:
            result = agent.receive_channel_message("cli", message, mgr)
        reply = result.get("reply")
        if reply:
            typer.echo(reply)
        if show_prompt_tokens and result.get("prompt_tokens") is not None:
            typer.echo(f"Prompt tokens: {result['prompt_tokens']}")

    return


@app.command("compress")
def compress_text(
    strategy: str = typer.Option(..., help="Compression strategy"),
    text: str = typer.Option(..., help="Text to compress"),
    budget: int = typer.Option(..., help="Token budget"),
    verbose_stats: bool = typer.Option(
        False, "--verbose-stats", help="Show token counts and processing time"
    ),
) -> None:
    """Run ``strategy`` on ``text`` with ``budget`` tokens."""

    try:
        strat_cls = get_compression_strategy(strategy)
    except KeyError:
        typer.secho(
            f"Unknown compression strategy '{strategy}'",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    strat = strat_cls()
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception:  # pragma: no cover - offline
        tokenizer = lambda t, **k: t.split()

    import time

    start = time.time()
    result = strat.compress(text, budget, tokenizer=tokenizer)
    if isinstance(result, tuple):
        compressed, trace = result
    else:
        compressed, trace = result, None
    elapsed = (time.time() - start) * 1000
    if trace and trace.processing_ms is None:
        trace.processing_ms = elapsed
    if verbose_stats:
        orig_tokens = token_count(tokenizer, text)
        comp_tokens = token_count(tokenizer, compressed.text)
        typer.echo(
            f"Original tokens: {orig_tokens}\nCompressed tokens: {comp_tokens}\nTime ms: {elapsed:.1f}"
        )
    typer.echo(compressed.text)


@app.command()
def validate(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
) -> None:
    """Validate the store metadata and embeddings."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        JsonNpyVectorStore(path=str(path))
    except EmbeddingDimensionMismatchError as exc:
        typer.secho(
            f"Embedding dimension mismatch: {exc}", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover - unexpected errors
        typer.secho(f"Error loading agent: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo("Store is valid")


@app.command()
def clear(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not delete files"),
) -> None:
    """Delete all data in the store."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if dry_run:
        store = JsonNpyVectorStore(path=str(path))
        typer.echo(
            f"Would delete {len(store.prototypes)} prototypes and {len(store.memories)} memories."
        )
        return
    if not yes:
        if not typer.confirm(
            f"Delete {path}?", abort=True
        ):  # pragma: no cover - user abort
            return
    if path.exists():
        shutil.rmtree(path)
        typer.echo(f"Deleted {path}")
    else:
        typer.secho("Directory not found", err=True, fg=typer.colors.RED)


@app.command("download-model")
def download_model(
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2", help="SentenceTransformer model name"
    ),
) -> None:
    """Pre-download a local embedding model."""
    from tqdm import tqdm
    from .model_utils import download_embedding_model

    bar = tqdm(total=1, desc="Downloading model", disable=False)
    download_embedding_model(model_name)
    bar.update(1)
    bar.close()
    typer.echo(f"Downloaded {model_name}")


@app.command("download-chat-model")
def download_chat_model(
    model_name: str = typer.Option("distilgpt2", help="Local causal LM name"),
) -> None:
    """Pre-download a local chat model for ``talk`` mode."""
    from tqdm import tqdm
    from .model_utils import download_chat_model as _download_chat_model

    bar = tqdm(total=1, desc="Downloading chat model", disable=False)
    _download_chat_model(model_name)
    bar.update(1)
    bar.close()
    typer.echo(f"Downloaded {model_name}")


@experiment_app.command("ingest")
def run_experiment_cmd(
    dataset: Path = typer.Argument(..., help="Text file to ingest"),
    work_dir: Optional[Path] = typer.Option(
        None, help="Directory for the temporary store"
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--tau", help="Similarity threshold"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run a simple ingestion experiment."""
    from .experiment_runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        dataset=dataset, similarity_threshold=similarity_threshold, work_dir=work_dir
    )
    metrics = run_experiment(cfg)
    if json_output:
        typer.echo(json.dumps(metrics))
    else:
        for k, v in metrics.items():
            typer.echo(f"{k}: {v}")


@package_app.command("create")
def package_create(
    name: str = typer.Option("sample_strategy", "--name", help="Strategy name"),
    path: Optional[Path] = typer.Option(None, "--path", help="Output directory"),
) -> None:
    """Generate a template strategy package."""
    target = Path(path or name)
    target.mkdir(parents=True, exist_ok=True)
    (target / "experiments").mkdir(exist_ok=True)
    (target / "strategy.py").write_text(
        """from gist_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace\n\n\nclass MyStrategy(CompressionStrategy):\n    id = \"{}\"\n\n    def compress(self, text_or_chunks, llm_token_budget, **kwargs):\n        return CompressedMemory(text=str(text_or_chunks)), CompressionTrace()\n""".format(
            name
        )
    )
    manifest = {
        "package_format_version": "1.0",
        "strategy_id": name,
        "strategy_class_name": "MyStrategy",
        "strategy_module": "strategy",
        "display_name": name,
        "version": "0.1.0",
        "authors": [],
        "description": "Describe the strategy",
    }
    import yaml

    (target / "strategy_package.yaml").write_text(yaml.safe_dump(manifest))
    (target / "requirements.txt").write_text("\n")
    (target / "README.md").write_text(f"# {name}\n")
    (target / "experiments" / "example.yaml").write_text(
        """dataset: example.txt\nparam_grid:\n- {}\npackaged_strategy_config:\n  strategy_params: {}\n"""
    )
    typer.echo(f"Created package at {target}")


@package_app.command("validate")
def package_validate(package_path: Path) -> None:
    """Validate a strategy package."""
    errors, warnings = validate_package_dir(package_path)
    for w in warnings:
        typer.secho(f"Warning: {w}", fg=typer.colors.YELLOW)
    if errors:
        for e in errors:
            typer.echo(e)
        raise typer.Exit(code=1)
    typer.echo("Package is valid")


@experiment_app.command("run-package")
def run_experiment_package(
    package_path: Path = typer.Argument(..., help="Path to strategy package"),
    experiment: Optional[str] = typer.Option(
        None, "--experiment", help="Experiment config"
    ),
) -> None:
    """Run an experiment from a strategy package."""
    manifest = load_manifest(package_path / "strategy_package.yaml")
    Strategy = load_strategy_class(package_path, manifest)

    missing = check_requirements_installed(package_path / "requirements.txt")
    if missing:
        typer.secho(
            f"Warning: missing requirements - {', '.join(missing)}",
            fg=typer.colors.YELLOW,
        )
    if experiment is None:
        defaults = manifest.get("default_experiments", [])
        if len(defaults) == 1:
            experiment = defaults[0].get("path")
    if experiment is None:
        typer.secho("No experiment specified", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    exp_path = Path(experiment)
    if not exp_path.is_absolute():
        exp_path = package_path / exp_path
    import yaml

    cfg_data = yaml.safe_load(exp_path.read_text()) or {}
    dataset = cfg_data.get("dataset")
    if dataset and not Path(dataset).is_absolute():
        cfg_data["dataset"] = str((package_path / dataset).resolve())
    params = cfg_data.get("param_grid", [{}])
    cfg = ResponseExperimentConfig(
        dataset=Path(cfg_data["dataset"]),
        param_grid=params,
        validation_metrics=cfg_data.get("validation_metrics"),
    )
    strategy_params = cfg_data.get("packaged_strategy_config", {}).get(
        "strategy_params", {}
    )
    strategy = Strategy(**strategy_params)
    results = run_response_experiment(cfg, strategy=strategy)
    typer.echo(json.dumps(results))


@experiment_app.command("optimize")
def optimize_experiment(
    script: Path = typer.Argument(..., help="Python script")
) -> None:
    """Run a hyperparameter optimisation script."""
    if not script.exists():
        typer.secho(f"Script '{script}' not found", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    import runpy

    runpy.run_path(str(script), run_name="__main__")


@trace_app.command("inspect")
def trace_inspect(
    trace_file: Path = typer.Argument(..., help="Path to trace JSON"),
    step_type: Optional[str] = typer.Option(None, "--type", help="Filter by step type"),
) -> None:
    """Print a summary of a saved ``CompressionTrace``."""

    if not trace_file.exists():
        typer.secho(
            f"Trace file '{trace_file}' not found", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    import json
    from rich.table import Table

    data = json.loads(trace_file.read_text())
    steps = data.get("steps", [])

    console.print(f"Strategy: {data.get('strategy_name', '')}")
    table = Table("idx", "type", "details")
    for idx, step in enumerate(steps):
        stype = step.get("type")
        if step_type and stype != step_type:
            continue
        preview = json.dumps(step.get("details", {}))[:50]
        table.add_row(str(idx), stype or "", preview)
    console.print(table)


if __name__ == "__main__":
    app()
