import json
import shutil
import os
import yaml
from pathlib import Path
from typing import Optional, Any # List removed
import logging
import time # Moved from _process_string_compression
from dataclasses import asdict # Moved from _process_string_compression
import sys # Moved from local imports
from tqdm import tqdm # Moved from download_model and download_chat_model
import runpy # Moved from optimize_experiment

import typer
import portalocker
from .token_utils import token_count
from rich.table import Table
from rich.console import Console

from gist_memory import __version__
from .logging_utils import configure_logging


from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from . import local_llm  # import module so tests can patch LocalChatModel
from .active_memory_manager import ActiveMemoryManager # Moved from talk
from .registry import _VALIDATION_METRIC_REGISTRY, get_validation_metric_class # Moved from metric_list, evaluate_compression, evaluate_llm_response
from . import llm_providers  # import module so tests can patch providers
from .model_utils import download_embedding_model, download_chat_model as _download_chat_model # Moved from download_model, download_chat_model
from .embedding_pipeline import (
    get_embedding_dim,
    EmbeddingDimensionMismatchError,
)
from .utils import load_agent
from .config import DEFAULT_BRAIN_PATH
from .compression import (
    available_strategies,
    get_compression_strategy,
    all_strategy_metadata,
)
from .plugin_loader import load_plugins
from .package_utils import (
    load_manifest,
    validate_manifest,
    load_strategy_class,
    validate_package_dir,
    check_requirements_installed,
)
from .response_experiment import ResponseExperimentConfig, run_response_experiment
from .experiment_runner import ExperimentConfig, run_experiment # Moved from run_experiment_cmd

app = typer.Typer(
    help="Gist Memory: A CLI for managing and interacting with a memory agent that uses advanced compression strategies to store and retrieve information. Primary actions like ingest, query, and talk are available as subcommands, along with advanced features for managing strategies, metrics, and experiments."
)
console = Console()
# VERBOSE = False # Unused global variable


def version_callback(value: bool):
    if value:
        typer.echo(f"Gist Memory version: {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Write debug logs to this file"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
    version: Optional[bool] = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Configure logging before executing commands."""
    if log_file:
        level = logging.DEBUG if verbose else logging.INFO
        configure_logging(log_file, level)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG)
    # global VERBOSE # Unused global variable
    # VERBOSE = verbose # Unused global variable
    load_plugins()
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
    """
    Creates and initializes a new agent data store at the specified directory.

    This command sets up the necessary file structure and metadata for a new
    Gist Memory agent. You can specify parameters like the agent name,
    embedding model, similarity threshold (tau), and chunking strategy.

    Usage Example:
        gist-memory init path/to/my_agent --model-name sentence-transformers/all-mpnet-base-v2 --tau 0.8
    """
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


strategy_app = typer.Typer(
    help="Manage and inspect Long-Term Memory (LTM) compression strategies."
)
app.add_typer(strategy_app, name="strategy")

metric_app = typer.Typer(
    help="Manage and inspect validation metrics for compression and LLM responses."
)
app.add_typer(metric_app, name="metric")

package_app = typer.Typer(help="Manage custom strategy packages.")
app.add_typer(package_app, name="package")

experiment_app = typer.Typer(
    help="Run various experiments for ingestion, responses, and hyperparameter optimization."
)
app.add_typer(experiment_app, name="experiment")

trace_app = typer.Typer(help="Inspect and analyze compression traces.")
app.add_typer(trace_app, name="trace")


@metric_app.command("list")
def metric_list() -> None:
    """
    Lists all available validation metric IDs that can be used with
    `evaluate-compression` and `evaluate-llm-response` commands.

    Usage Example:
        gist-memory metric list
    """
    # from .registry import _VALIDATION_METRIC_REGISTRY # Moved to top

    for mid in sorted(_VALIDATION_METRIC_REGISTRY):
        typer.echo(mid)


@strategy_app.command("inspect")
def strategy_inspect(
    strategy_name: str,
    *,
    agent_name: str = typer.Option(DEFAULT_BRAIN_PATH, help="Agent directory"),
    list_prototypes: bool = typer.Option(
        False, "--list-prototypes", help="List prototypes for the strategy"
    ),
) -> None:
    """
    Inspect the details of a compression strategy, particularly its prototypes.

    This command allows you to inspect the details of a compression strategy.
    For the 'prototype' strategy, you can list the learned prototypes, which are
    representative summaries of related memories within the specified agent.

    Usage Example:
        gist-memory strategy inspect prototype --agent-name path/to/agent --list-prototypes
    """

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


@strategy_app.command("list")
def strategy_list() -> None:
    """
    Lists all available Long-Term Memory (LTM) compression strategies
    along with their metadata such as display name, version, and source.

    Usage Example:
        gist-memory strategy list
    """
    load_plugins()
    table = Table("Strategy ID", "Display Name", "Version", "Source", "Status")
    meta = all_strategy_metadata()
    for sid in available_strategies():
        info = meta.get(sid, {})
        status = ""
        if info.get("overrides"):
            status = f"Overrides {info['overrides']}"
        table.add_row(
            sid,
            info.get("display_name", sid) or sid,
            info.get("version", "N/A") or "N/A",
            info.get("source", "built-in") or "built-in",
            status,
        )
    console.print(table)


@app.command()
def stats(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """
    Show statistics about the agent's data store.

    This command displays various statistics about the specified agent's memory,
    such as the number of prototypes and memories.
    Use the --json option to dump these statistics in a machine-readable JSON format,
    which can be useful for scripting or further analysis.

    Usage Example (dump to JSON and redirect to a file):
        gist-memory stats path/to/agent --json > agent_stats.json
    """
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
    model_name: str = typer.Option("tiny-gpt2", help="Local chat model"),
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
    """
    Interact with the memory agent by sending it a message.

    This command allows you to have a conversation with the agent. The agent uses its
    stored knowledge, organized into prototypes, to understand and respond to your messages.

    Prototypes are representative summaries of related memories. When you talk to the
    agent, it uses these prototypes to retrieve relevant information to formulate a response.

    Usage Example:
        gist-memory talk --agent-name path/to/agent --message "What do you know about topic X?"
    """

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

        # from .local_llm import LocalChatModel # Moved to top

        try:
            agent._chat_model = local_llm.LocalChatModel(model_name=model_name)
            agent._chat_model.load_model()
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)

        # from .active_memory_manager import ActiveMemoryManager # Moved to top

        mgr = ActiveMemoryManager()
        comp = None
        if compression_strategy: # Simplified condition
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


def _process_string_compression(
    text_content: str,
    strategy_id: str,
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
) -> None:
    """Compress a string and handle output."""
    try:
        strat_cls = get_compression_strategy(strategy_id)
    except KeyError:
        typer.secho(
            f"Unknown compression strategy '{strategy_id}'. Available: {', '.join(available_strategies())}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    strat = strat_cls()
    # import time # Moved to top

    start = time.time()
    result = strat.compress(text_content, budget, tokenizer=tokenizer)
    if isinstance(result, tuple):
        compressed, trace = result
    else:
        compressed, trace = result, None
    elapsed = (time.time() - start) * 1000
    if trace and trace.processing_ms is None:
        trace.processing_ms = elapsed
    if verbose_stats:
        orig_tokens = token_count(tokenizer, text_content)
        comp_tokens = token_count(tokenizer, compressed.text)
        typer.echo(
            f"Original tokens: {orig_tokens}\nCompressed tokens: {comp_tokens}\nTime ms: {elapsed:.1f}"
        )
    if output_file:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(compressed.text)
        except (IOError, OSError, PermissionError) as exc:
            typer.secho(f"Error writing {output_file}: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"Saved compressed output to {output_file}")
    else:
        typer.echo(compressed.text)

    if trace_file and trace:
        # from dataclasses import asdict # Moved to top

        try:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(json.dumps(asdict(trace))) # json is imported globally
        except (IOError, OSError, PermissionError) as exc:
            typer.secho(f"Error writing {trace_file}: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)


def _process_file_compression(
    file_path: Path,
    strategy_id: str,
    budget: int,
    output_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    trace_file: Optional[Path],
) -> None:
    """Compress text loaded from ``file_path``."""
    if not file_path.exists() or not file_path.is_file():
        typer.secho(f"File not found: {file_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        text_content = file_path.read_text()
    except (IOError, OSError, PermissionError) as exc:
        typer.secho(f"Error reading {file_path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _process_string_compression(text_content, strategy_id, budget, output_file, trace_file, verbose_stats, tokenizer)


def _process_directory_compression(
    dir_path: Path,
    strategy_id: str,
    budget: int,
    output_dir_param: Optional[Path],
    recursive: bool,
    pattern: str,
    verbose_stats: bool,
    tokenizer: Any,
    trace_file: Optional[Path],
) -> None:
    """Compress all files in ``dir_path`` matching ``pattern``."""
    if not dir_path.exists() or not dir_path.is_dir():
        typer.secho(f"Directory not found: {dir_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    files = list(dir_path.rglob(pattern) if recursive else dir_path.glob(pattern))
    if not files:
        typer.echo("No matching files found.")
        return
    if trace_file is not None:
        typer.secho("--output-trace ignored for directory input", fg=typer.colors.YELLOW)
        trace_file = None

    count = 0
    for input_file in files:
        typer.echo(f"Processing {input_file}...")
        if output_dir_param:
            try:
                output_dir_param.mkdir(parents=True, exist_ok=True)
            except (IOError, OSError, PermissionError) as exc:
                typer.secho(f"Error creating {output_dir_param}: {exc}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            rel = input_file.relative_to(dir_path)
            out_path = output_dir_param / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = input_file.with_name(f"{input_file.stem}_compressed{input_file.suffix}")

        _process_file_compression(
            input_file, strategy_id, budget, out_path, verbose_stats, tokenizer, None
        )
        count += 1

    if verbose_stats:
        typer.echo(f"Processed {count} files.")


@app.command("compress")
def compress_text(
    input_source: str = typer.Argument(
        ...,
        help="Input text directly or path to a file or directory to compress",
    ),
    *,
    strategy: str = typer.Option(..., help="Compression strategy"),
    output_path: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        resolve_path=True,
        help=(
            "Path to write compressed output. For directories, this specifies the"
            " root output directory."
        ),
    ),
    output_trace: Optional[Path] = typer.Option(
        None,
        "--output-trace",
        resolve_path=True,
        help="Write CompressionTrace JSON to this file",
    ),
    recursive: bool = typer.Option(
        False,
        "-r",
        "--recursive",
        help="Process directories recursively",
    ),
    pattern: str = typer.Option(
        "*.txt",
        "-p",
        "--pattern",
        help="File glob pattern when reading from a directory",
    ),
    budget: int = typer.Option(..., help="Token budget"),
    verbose_stats: bool = typer.Option(
        False, "--verbose-stats", help="Show token counts and processing time"
    ),
) -> None:
    """
    Compress input text, file, or directory using a specified strategy to create a summary.

    This command takes an input string, a single file, or all files in a directory (optionally recursively)
    and compresses the content using the chosen strategy and token budget.
    The result is a condensed version (summary) of the input.

    Use 'gist-memory strategy list' to see available strategy IDs.

    Usage Example:
        gist-memory compress "This is a long text that needs to be summarized effectively." --strategy prototype --budget 50
        gist-memory compress path/to/document.txt --strategy my_custom_strategy --budget 200 -o path/to/summary.txt
    """

    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        def tokenizer(text: str, return_tensors=None) -> dict:
            return {"input_ids": enc.encode(text)}

    except Exception:  # pragma: no cover - offline
        tokenizer = lambda t, **k: t.split()

    source_as_path = Path(input_source)

    if source_as_path.is_file():
        if recursive or pattern != "*.txt":
            if recursive:
                typer.secho("--recursive ignored for file input", fg=typer.colors.YELLOW)
            if pattern != "*.txt":
                typer.secho("--pattern ignored for file input", fg=typer.colors.YELLOW)
        _process_file_compression(
            source_as_path,
            strategy,
            budget,
            output_path,
            verbose_stats,
            tokenizer,
            output_trace,
        )
    elif source_as_path.is_dir():
        if "**" in pattern and not recursive:
            typer.secho(
                "Pattern includes '**' but --recursive not set; matching may miss subdirectories",
                fg=typer.colors.YELLOW,
            )
        _process_directory_compression(
            source_as_path,
            strategy,
            budget,
            output_path,
            recursive,
            pattern,
            verbose_stats,
            tokenizer,
            output_trace,
        )
    else:
        if recursive or pattern != "*.txt":
            if recursive:
                typer.secho("--recursive ignored for string input", fg=typer.colors.YELLOW)
            if pattern != "*.txt":
                typer.secho("--pattern ignored for string input", fg=typer.colors.YELLOW)
        _process_string_compression(
            input_source,
            strategy,
            budget,
            output_path,
            output_trace,
            verbose_stats,
            tokenizer,
        )


@app.command("evaluate-compression")
def evaluate_compression(
    original_input: str = typer.Argument(..., help="Original text or file path"),
    compressed_input: str = typer.Argument(..., help="Compressed text or file path, or '-' for stdin"),
    metric_id: str = typer.Option(..., "--metric", "-m", help="Metric ID"),
    metric_params_json: Optional[str] = typer.Option(None, "--metric-params", help="Metric parameters as JSON"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """
    Evaluates the quality of compressed text against the original using a specified metric.

    This command helps quantify how well the compression performed based on criteria
    like information retention, size reduction, etc. Use `gist-memory metric list`
    to see available metric IDs.

    Usage Example:
        gist-memory evaluate-compression "Original long text..." "Compressed summary." --metric compression_ratio
    """
    # import sys # Moved to top
    # import json # Global import
    # from .registry import get_validation_metric_class # Moved to top

    def read_input(value: str, allow_stdin: bool) -> str:
        if value == "-":
            if not allow_stdin:
                typer.secho("stdin already used", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(value)
        if p.exists() and p.is_file():
            return p.read_text()
        return value

    orig_text = read_input(original_input, True)
    comp_text = read_input(compressed_input, compressed_input == "-")

    try:
        Metric = get_validation_metric_class(metric_id)
    except KeyError:
        typer.secho(f"Unknown metric '{metric_id}'", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    params = {}
    if metric_params_json:
        try:
            params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc:
            typer.secho(f"Invalid metric params: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    metric = Metric(**params)
    try:
        scores = metric.evaluate(original_text=orig_text, compressed_text=comp_text)
    except Exception as exc:
        typer.secho(f"Metric error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(scores))
    else:
        for k, v in scores.items():
            typer.echo(f"{k}: {v}")


@app.command("llm-prompt")
def llm_prompt(
    *,
    context_input: str = typer.Option(..., "--context", "-c", help="Compressed context string, path, or '-' for stdin"),
    query: str = typer.Option(..., "--query", "-q", help="User query"),
    model_id: str = typer.Option("tiny-gpt2", "--model", help="Model ID"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", "-s", help="Optional system prompt"),
    max_new_tokens: int = typer.Option(150, help="Max new tokens"),
    output_llm_response_file: Optional[Path] = typer.Option(None, "--output-response", help="File to save response"),
    llm_config_file: Optional[Path] = typer.Option(Path("llm_models_config.yaml"), "--llm-config", exists=True, dir_okay=False, resolve_path=True, help="LLM config file"),
    api_key_env_var: Optional[str] = typer.Option(None, help="Env var for API key"),
) -> None:
    """
    Sends a constructed prompt (context + query) to a specified Language Model (LLM).

    This command is useful for testing how an LLM responds when provided with
    context that has been compressed by one of the strategies. You can specify
    the model, system prompt, and other parameters.

    Usage Example:
        gist-memory llm-prompt --context "My compressed context about AI." --query "What are the key points?" --model-id gpt-3.5-turbo
    """
    # import sys # Moved to top
    # import json # Global import
    # import yaml # Global import
    # from .llm_providers import OpenAIProvider, GeminiProvider, LocalTransformersProvider # Moved to top

    def read_val(val: str) -> str:
        if val == "-":
            return sys.stdin.read()
        p = Path(val)
        if p.exists() and p.is_file():
            return p.read_text()
        return val

    context_text = read_val(context_input)

    cfg = {}
    if llm_config_file:
        try:
            cfg = yaml.safe_load(llm_config_file.read_text()) or {}
        except Exception as exc:
            typer.secho(f"Error loading config: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    model_cfg = cfg.get(model_id, {"provider": "local", "model_name": model_id})
    provider_name = model_cfg.get("provider")
    model_name = model_cfg.get("model_name", model_id)

    if provider_name == "openai":
        provider = llm_providers.OpenAIProvider()
    elif provider_name == "gemini":
        provider = llm_providers.GeminiProvider()
    else:
        provider = llm_providers.LocalTransformersProvider()

    api_key = os.getenv(api_key_env_var) if api_key_env_var else None

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    prompt_parts.append(context_text)
    prompt_parts.append(query)
    prompt = "\n".join(part for part in prompt_parts if part)

    try:
        response = provider.generate_response(prompt, model_name=model_name, max_new_tokens=max_new_tokens, api_key=api_key)
    except Exception as exc:
        typer.secho(f"LLM error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if output_llm_response_file:
        try:
            output_llm_response_file.write_text(response)
        except Exception as exc:
            typer.secho(f"Error writing {output_llm_response_file}: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
    else:
        typer.echo(response)


@app.command("evaluate-llm-response")
def evaluate_llm_response(
    response_input: str = typer.Argument(..., help="LLM response text or file, or '-' for stdin"),
    reference_input: str = typer.Argument(..., help="Reference answer text or file"),
    metric_id: str = typer.Option(..., "--metric", "-m", help="Metric ID"),
    metric_params_json: Optional[str] = typer.Option(None, "--metric-params", help="Metric parameters as JSON"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """
    Evaluates an LLM's response against a reference (ground truth) answer using a specified metric.

    This helps in assessing the performance of an LLM in tasks like question answering
    or summarization when its responses are compared to known good answers.
    Use `gist-memory metric list` to see available metric IDs.

    Usage Example:
        gist-memory evaluate-llm-response "LLM Output: The capital is Paris." "Reference Answer: Paris." --metric exact_match
    """
    # import sys # Moved to top
    # import json # Global import
    # from .registry import get_validation_metric_class # Moved to top

    def read_value(val: str, allow_stdin: bool) -> str:
        if val == "-":
            if not allow_stdin:
                typer.secho("stdin already used", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(val)
        if p.exists() and p.is_file():
            return p.read_text()
        return val

    resp_text = read_value(response_input, True)
    ref_text = read_value(reference_input, False)

    try:
        Metric = get_validation_metric_class(metric_id)
    except KeyError:
        typer.secho(f"Unknown metric '{metric_id}'", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    params = {}
    if metric_params_json:
        try:
            params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc:
            typer.secho(f"Invalid metric params: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    metric = Metric(**params)
    try:
        scores = metric.evaluate(llm_response=resp_text, reference_answer=ref_text)
    except Exception as exc:
        typer.secho(f"Metric error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(scores))
    else:
        for k, v in scores.items():
            typer.echo(f"{k}: {v}")


@app.command()
def validate(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
) -> None:
    """
    Checks the integrity and validity of an existing agent data store.

    This command verifies that the agent's metadata, embedding dimensions,
    and file structures are correct and consistent. It's useful for diagnosing
    potential issues with an agent store.

    Usage Example:
        gist-memory validate path/to/my_agent
    """
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
    """
    Deletes all data within the specified agent store, including memories and prototypes.

    This is a destructive operation and requires confirmation via the --yes flag
    unless --dry-run is used. Use with caution.

    Usage Example (will prompt for confirmation):
        gist-memory clear path/to/my_agent

    Usage Example (will not prompt, immediately deletes):
        gist-memory clear path/to/my_agent --yes
    """
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
    """
    Pre-downloads a specified sentence transformer model for generating embeddings.

    This is useful to ensure the model is available locally, especially in
    environments with limited internet access or to avoid download delays
    during agent initialization or operation.

    Usage Example:
        gist-memory download-model --model-name sentence-transformers/all-MiniLM-L12-v2
    """
    # from tqdm import tqdm # Moved to top
    # from .model_utils import download_embedding_model # Moved to top

    bar = tqdm(total=1, desc="Downloading model", disable=False)
    download_embedding_model(model_name)
    bar.update(1)
    bar.close()
    typer.echo(f"Downloaded {model_name}")


@app.command("download-chat-model")
def download_chat_model(
    model_name: str = typer.Option("tiny-gpt2", help="Local causal LM name"),
) -> None:
    """
    Pre-downloads a specified local language model for use with the `talk` command.

    This ensures the chat model is available locally, which can speed up
    the `talk` command's first run and is helpful for offline usage.

    Usage Example:
        gist-memory download-chat-model --model-name tiny-gpt2
    """
    # from tqdm import tqdm # Moved to top
    # from .model_utils import download_chat_model as _download_chat_model # Moved to top

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
    """
    Run a simple ingestion experiment.

    This command processes a dataset (e.g., a text file) to build or update the agent's memory.
    During ingestion, new information is chunked and then either assigned to an existing
    prototype or used to create a new one based on similarity.

    Prototypes are representative summaries of related memories.

    Usage Example:
        gist-memory experiment ingest path/to/dataset.txt --tau 0.75
    """
    # from .experiment_runner import ExperimentConfig, run_experiment # Moved to top

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
    """
    Generates a template for creating a new custom strategy package.

    This command creates a directory with the basic structure and files needed
    to develop and package a new compression strategy.

    Usage Example:
        gist-memory package create --name my_new_strategy
    """
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
    # import yaml # Global import

    (target / "strategy_package.yaml").write_text(yaml.safe_dump(manifest))
    (target / "requirements.txt").write_text("\n")
    (target / "README.md").write_text(f"# {name}\n")
    (target / "experiments" / "example.yaml").write_text(
        """dataset: example.txt\nparam_grid:\n- {}\npackaged_strategy_config:\n  strategy_params: {}\n"""
    )
    typer.echo(f"Created package at {target}")


@package_app.command("validate")
def package_validate(package_path: Path) -> None:
    """
    Validates the structure, manifest file (strategy_package.yaml), and other
    requirements for a custom strategy package.

    This helps ensure the package is correctly formatted before sharing or use.

    Usage Example:
        gist-memory package validate path/to/my_package
    """
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
    """
    Runs an experiment defined within a specified strategy package.

    Experiments are typically defined in YAML files within the package and
    can be used to evaluate the packaged strategy's performance.

    Usage Example:
        gist-memory experiment run-package path/to/my_package --experiment experiment_config.yaml
    """
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
    # import yaml # Global import

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
    """
    Runs a hyperparameter optimization script for finding optimal strategy parameters.

    The script should use a library like Optuna or Ray Tune to explore
    different parameter combinations for a compression strategy.

    Usage Example:
        gist-memory experiment optimize path/to/optimization_script.py
    """
    if not script.exists():
        typer.secho(f"Script '{script}' not found", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # import runpy # Moved to top

    runpy.run_path(str(script), run_name="__main__")


@trace_app.command("inspect")
def trace_inspect(
    trace_file: Path = typer.Argument(..., help="Path to trace JSON"),
    step_type: Optional[str] = typer.Option(None, "--type", help="Filter by step type"),
) -> None:
    """
    Prints a summary of a saved CompressionTrace JSON file.

    This command helps in analyzing the steps taken by a compression strategy.
    You can filter the displayed steps by their type (e.g., 'chunk', 'embed', 'cluster').

    Usage Example:
        gist-memory trace inspect path/to/trace.json --type filter_item
    """

    if not trace_file.exists():
        typer.secho(
            f"Trace file '{trace_file}' not found", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # import json # Global import
    # from rich.table import Table # Global import

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
