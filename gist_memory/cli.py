import json
import shutil
import os
import yaml
from pathlib import Path
from typing import Optional, Any
import logging
import time
from dataclasses import asdict
import sys
from tqdm import tqdm
import runpy

import typer
import portalocker
from .token_utils import token_count
from rich.table import Table
from rich.console import Console

from gist_memory import __version__
from .logging_utils import configure_logging


from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from . import local_llm
from .active_memory_manager import ActiveMemoryManager
from .registry import (
    _VALIDATION_METRIC_REGISTRY,
    get_validation_metric_class,
)
from . import llm_providers
from .model_utils import (
    download_embedding_model as util_download_embedding_model,
    download_chat_model as util_download_chat_model,
)
from .embedding_pipeline import (
    get_embedding_dim,
    EmbeddingDimensionMismatchError,
)
from .utils import load_agent
from gist_memory.config import Config

from .compression import (
    available_strategies,
    get_compression_strategy,
    all_strategy_metadata,
)
from .plugin_loader import load_plugins
from .cli_plugins import load_cli_plugins
from .package_utils import (
    load_manifest,
    validate_manifest,
    load_strategy_class,
    validate_package_dir,
    check_requirements_installed,
)
from .response_experiment import ResponseExperimentConfig, run_response_experiment
from .experiment_runner import (
    ExperimentConfig,
    run_experiment,
)

app = typer.Typer(
    help="Gist Memory: A CLI for managing and interacting with a memory agent that uses advanced compression strategies to store and retrieve information. Primary actions like ingest, query, and talk are available as subcommands, along with advanced features for managing strategies, metrics, and experiments."
)
console = Console()

# --- New Command Groups ---
agent_app = typer.Typer(help="Manage agents: initialize, inspect statistics, validate, and clear.")
config_app = typer.Typer(help="Manage Gist Memory configuration settings.")
dev_app = typer.Typer(help="Developer tools for testing, evaluation, and package management.")

# --- Add New Command Groups to Main App ---
app.add_typer(agent_app, name="agent")
app.add_typer(config_app, name="config")
app.add_typer(dev_app, name="dev")

def version_callback(value: bool):
    if value:
        typer.echo(f"Gist Memory version: {__version__}")
        raise typer.Exit()

@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Write debug logs to this file"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Enable debug logging"),
    memory_path: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Memory store directory (overrides config/env)"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Default model ID for LLM interactions (overrides config/env)"),
    strategy_id: Optional[str] = typer.Option(None, "--strategy-id", help="Default compression strategy ID (overrides config/env)"),
    version: Optional[bool] = typer.Option(None, "-v", "--version", callback=version_callback, is_eager=True, help="Show the version and exit."),
) -> None:
    config = Config()
    config.validate()

    resolved_log_file = log_file
    resolved_verbose = verbose

    if resolved_log_file:
        level = logging.DEBUG if resolved_verbose else logging.INFO
        configure_logging(resolved_log_file, level)
    elif resolved_verbose:
        logging.basicConfig(level=logging.DEBUG)

    resolved_memory_path = memory_path if memory_path is not None else config.get("gist_memory_path")
    resolved_model_id = model_id if model_id is not None else config.get("default_model_id")
    resolved_strategy_id = strategy_id if strategy_id is not None else config.get("default_strategy_id")

    if resolved_memory_path:
        resolved_memory_path = str(Path(resolved_memory_path).expanduser())

    # Determine if the command is one that requires memory_path to be set
    command_requires_memory_path = True # Assume true by default
    if ctx.invoked_subcommand:
        # Commands that DON'T require a memory path immediately
        no_mem_path_commands = ["config", "dev"]
        # Top level commands that might also not need it (e.g. if they are just info commands)
        # For now, assume all other top-level commands (like ingest, query, summarize) will need it.
        if ctx.invoked_subcommand in no_mem_path_commands:
            command_requires_memory_path = False
        else: # Check if it's a subcommand of a group that doesn't need it
            parent_command = ctx.invoked_subcommand.split('.')[0] # Simplistic check, assumes one level of nesting
            if parent_command in no_mem_path_commands:
                 command_requires_memory_path = False


    if command_requires_memory_path and not resolved_memory_path :
        is_interactive = sys.stdin.isatty()
        prompt_default_path = config.get("gist_memory_path")

        if is_interactive:
            typer.secho("The Gist Memory path is not set.", fg=typer.colors.YELLOW)
            new_path_input = typer.prompt("Please enter the path for Gist Memory storage", default=str(Path(prompt_default_path).expanduser()) if prompt_default_path else None)
            if new_path_input:
                resolved_memory_path = str(Path(new_path_input).expanduser())
                typer.secho(f"Using memory path: {resolved_memory_path}", fg=typer.colors.GREEN)
                typer.echo(f"To set this path permanently, run: gist-memory config set gist_memory_path \"{resolved_memory_path}\"")
            else:
                typer.secho("Memory path is required to proceed.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
        else:
            typer.secho("Error: Gist Memory path is not set.", fg=typer.colors.RED, err=True)
            typer.secho("Please set it using the --memory-path option, the GIST_MEMORY_PATH environment variable, or in a config file (~/.config/gist_memory/config.yaml or .gmconfig.yaml).", err=True)
            raise typer.Exit(code=1)

    load_plugins()

    if ctx.obj is None: ctx.obj = {}
    ctx.obj.update({
        "verbose": resolved_verbose, "log_file": resolved_log_file,
        "gist_memory_path": resolved_memory_path, "default_model_id": resolved_model_id,
        "default_strategy_id": resolved_strategy_id, "config": config,
    })

class PersistenceLock:
    def __init__(self, path: Path) -> None: self.file = (path / ".lock").open("a+")
    def __enter__(self): portalocker.lock(self.file, portalocker.LockFlags.EXCLUSIVE); return self
    def __exit__(self, exc_type, exc, tb): portalocker.unlock(self.file); self.file.close()

def _corrupt_exit(path: Path, exc: Exception) -> None:
    typer.echo(f"Error: Brain data is corrupted. {exc}", err=True)
    typer.echo(f"Try running gist-memory validate {path} for more details or restore from a backup.", err=True)
    raise typer.Exit(code=1)

def _load_agent(path: Path) -> Agent:
    try: return load_agent(path)
    except Exception as exc: _corrupt_exit(path, exc)

# --- Agent Commands ---
@agent_app.command("init")
def init(target_directory: Path = typer.Argument(..., help="Target directory for the new agent. This will be created if it doesn't exist.", resolve_path=True ), *, ctx: typer.Context, name: str = typer.Option("default", help="Name of the agent"), model_name: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name"), tau: float = 0.8, alpha: float = typer.Option(0.1, help="Alpha parameter for the agent"), chunker: str = typer.Option("sentence_window", help="Chunking strategy for the agent")) -> None:
    path = target_directory.expanduser()
    if path.exists() and any(path.iterdir()): typer.secho("Directory already exists and is not empty", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    if not 0.5 <= tau <= 0.95: typer.secho("Error: --tau must be between 0.5 and 0.95.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    try: dim = get_embedding_dim()
    except RuntimeError as exc: typer.echo(str(exc), err=True); raise typer.Exit(code=1)
    store = JsonNpyVectorStore(path=str(path), embedding_model=model_name, embedding_dim=dim)
    store.meta.update({"agent_name": name, "tau": tau, "alpha": alpha, "chunker": chunker})
    store.save(); typer.echo(f"Initialized agent at {path}")

@agent_app.command("stats")
def stats(ctx: typer.Context, memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Memory store directory (if different from global --memory-path)"), json_output: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for stats.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    agent = _load_agent(path); data = agent.get_statistics(); logging.debug("Collected statistics: %s", data)
    if json_output: typer.echo(json.dumps(data))
    else:
        for k, v in data.items(): typer.echo(f"{k}: {v}")

@agent_app.command("validate")
def validate_agent_storage(ctx: typer.Context, memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Memory store directory (if different from global --memory-path)")) -> None:
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for validate.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    try: JsonNpyVectorStore(path=str(path))
    except EmbeddingDimensionMismatchError as exc: typer.secho(f"Embedding dimension mismatch: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    except Exception as exc: typer.secho(f"Error loading agent: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    typer.echo("Store is valid")

@agent_app.command("clear")
def clear(ctx: typer.Context, memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Memory store directory (if different from global --memory-path)"), force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"), dry_run: bool = typer.Option(False, "--dry-run", help="Do not delete files")) -> None:
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for clear.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    if dry_run: store = JsonNpyVectorStore(path=str(path)); typer.echo(f"Would delete {len(store.prototypes)} prototypes and {len(store.memories)} memories."); return
    if not force:
        if not typer.confirm(f"Delete {path}?", abort=True): return
    if path.exists(): shutil.rmtree(path); typer.echo(f"Deleted {path}")
    else: typer.secho("Directory not found", err=True, fg=typer.colors.RED)

# --- Top-Level Commands ---
@app.command("ingest", help="Ingests a text file or files in a directory into the agent's memory.")
def ingest(
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="Path to the text file or directory to ingest.", exists=True, file_okay=True, dir_okay=True, resolve_path=True),
    tau: Optional[float] = typer.Option(None, "--tau", "-t", help="Similarity threshold for memory consolidation. Overrides agent's existing tau. If path is new, this tau is used."),
    json_output: bool = typer.Option(False, "--json", help="Output ingestion statistics as JSON.")
) -> None:
    """
    Ingests text from a file or directory into the agent's memory specified by the global --memory-path.
    """
    memory_path_str = ctx.obj.get("gist_memory_path")
    if not memory_path_str:
        typer.secho("Error: Memory path not set. Use --memory-path or set in config.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    agent_path = Path(memory_path_str)
    final_tau = tau # Use CLI tau if provided

    if final_tau is None: # If no CLI tau, try to load from existing agent or use default
        if agent_path.exists() and (agent_path / "meta.json").is_file():
            try:
                with open(agent_path / "meta.json", "r") as f:
                    meta_data = json.load(f)
                agent_default_tau = meta_data.get("tau")
                if agent_default_tau is not None:
                    final_tau = agent_default_tau
                else:
                    final_tau = 0.8 # Default if not in meta
                if tau is not None: # CLI tau overrides agent's tau
                    final_tau = tau
                    typer.echo(f"Using provided tau: {final_tau} (overriding agent's tau: {agent_default_tau or 'Not Set'})")
                else:
                    typer.echo(f"Using agent's existing tau: {final_tau}")
            except Exception as e:
                typer.secho(f"Could not load agent metadata to determine tau: {e}. Using default 0.8.", fg=typer.colors.YELLOW)
                final_tau = 0.8
        else:
            final_tau = 0.8 # Default for new agent
            typer.echo(f"No agent found at {agent_path}. Initializing with tau: {final_tau}")
    elif tau is not None: # CLI tau provided
         typer.echo(f"Using provided tau for ingestion: {final_tau}")


    # ExperimentConfig expects 'dataset', 'similarity_threshold', and 'work_dir'.
    # We use 'source' for dataset, 'final_tau' for similarity_threshold,
    # and the global 'agent_path' as the 'work_dir' for ingestion.
    cfg = ExperimentConfig(
        dataset=source,  # source is a Path object from Typer
        similarity_threshold=final_tau,
        work_dir=agent_path  # Ingest directly into the specified agent directory
    )

    typer.echo(f"Ingesting data from '{source}' into agent at '{agent_path}' with tau={final_tau}...")
    try:
        metrics = run_experiment(cfg)
        if json_output:
            typer.echo(json.dumps(metrics))
        else:
            for k, v_val in metrics.items(): # renamed v to v_val
                typer.echo(f"{k}: {v_val}")
        typer.echo("Ingestion complete.")
    except Exception as e:
        typer.secho(f"Error during ingestion: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@app.command("query", help="Queries the agent with the provided text and returns a response.")
def query(ctx: typer.Context, query_text: str = typer.Argument(..., help="Query to ask the agent."), show_prompt_tokens: bool = typer.Option(False, "--show-prompt-tokens", help="Display token count of the prompt sent to the LLM")) -> None:
    final_memory_path_str = ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for query.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    with PersistenceLock(path):
        agent = _load_agent(path); final_model_name = ctx.obj.get("default_model_id"); final_compression_strategy = ctx.obj.get("default_strategy_id")
        if final_model_name is None: typer.secho("Error: Model ID not specified globally or via config.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
        try: agent._chat_model = local_llm.LocalChatModel(model_name=final_model_name); agent._chat_model.load_model()
        except RuntimeError as exc: typer.echo(str(exc), err=True); raise typer.Exit(code=1)
        mgr = ActiveMemoryManager(); comp = None
        if final_compression_strategy and final_compression_strategy.lower() != "none":
            try: comp_cls = get_compression_strategy(final_compression_strategy); comp = comp_cls()
            except KeyError: typer.secho(f"Unknown compression strategy '{final_compression_strategy}' (from global config).", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
        try: result = agent.receive_channel_message("cli", query_text, mgr, compression=comp)
        except TypeError: result = agent.receive_channel_message("cli", query_text, mgr)
        reply = result.get("reply")
        if reply: typer.echo(reply)
        if show_prompt_tokens and result.get("prompt_tokens") is not None: typer.echo(f"Prompt tokens: {result['prompt_tokens']}")

@app.command("summarize", help="Summarizes the given text content or file/directory using a specified strategy.")
def summarize(ctx: typer.Context, input_source: str = typer.Argument(..., help="Input text directly or path to a file or directory to summarize"), *, strategy_arg: Optional[str] = typer.Option(None, "--strategy", "-s", help="Compression strategy (overrides global default_strategy_id)"), output_path: Optional[Path] = typer.Option(None, "-o", "--output", resolve_path=True, help=("Path to write compressed output. For directories, this specifies the root output directory.")), output_trace: Optional[Path] = typer.Option(None, "--output-trace", resolve_path=True, help="Write CompressionTrace JSON to this file"), recursive: bool = typer.Option(False, "-r", "--recursive", help="Process directories recursively"), pattern: str = typer.Option("*.txt", "-p", "--pattern", help="File glob pattern when reading from a directory"), budget: int = typer.Option(..., help="Token budget"), verbose_stats: bool = typer.Option(False, "--verbose-stats", help="Show token counts and processing time")) -> None:
    final_strategy_id = strategy_arg if strategy_arg is not None else ctx.obj.get("default_strategy_id")
    if not final_strategy_id: typer.secho("Error: Compression strategy not specified via --strategy or global/config default_strategy_id.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    try: import tiktoken; enc = tiktoken.get_encoding("gpt2"); tokenizer = lambda text, **k: {"input_ids": enc.encode(text)}
    except Exception: tokenizer = lambda t, **k: t.split()
    source_as_path = Path(input_source)
    if source_as_path.is_file():
        if recursive or pattern != "*.txt":
            if recursive: typer.secho("--recursive ignored for file input", fg=typer.colors.YELLOW)
            if pattern != "*.txt": typer.secho("--pattern ignored for file input", fg=typer.colors.YELLOW)
        _process_file_compression(source_as_path, final_strategy_id, budget, output_path, verbose_stats, tokenizer, output_trace)
    elif source_as_path.is_dir():
        if "**" in pattern and not recursive: typer.secho("Pattern includes '**' but --recursive not set; matching may miss subdirectories", fg=typer.colors.YELLOW)
        _process_directory_compression(source_as_path, final_strategy_id, budget, output_path, recursive, pattern, verbose_stats, tokenizer, output_trace)
    else:
        if recursive or pattern != "*.txt":
            if recursive: typer.secho("--recursive ignored for string input", fg=typer.colors.YELLOW)
            if pattern != "*.txt": typer.secho("--pattern ignored for string input", fg=typer.colors.YELLOW)
        _process_string_compression(input_source, final_strategy_id, budget, output_path, output_trace, verbose_stats, tokenizer)

# --- Helper functions for summarize (formerly compress_text) ---
def _process_string_compression(text_content: str, strategy_id: str, budget: int, output_file: Optional[Path], trace_file: Optional[Path], verbose_stats: bool, tokenizer: Any) -> None:
    try: strat_cls = get_compression_strategy(strategy_id)
    except KeyError: typer.secho(f"Unknown compression strategy '{strategy_id}'. Available: {', '.join(available_strategies())}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    strat = strat_cls(); start = time.time(); result = strat.compress(text_content, budget, tokenizer=tokenizer)
    if isinstance(result, tuple): compressed, trace = result
    else: compressed, trace = result, None
    elapsed = (time.time() - start) * 1000
    if trace and trace.processing_ms is None: trace.processing_ms = elapsed
    if verbose_stats:
        orig_tokens = token_count(tokenizer, text_content); comp_tokens = token_count(tokenizer, compressed.text)
        typer.echo(f"Original tokens: {orig_tokens}\nCompressed tokens: {comp_tokens}\nTime ms: {elapsed:.1f}")
    if output_file:
        try: output_file.parent.mkdir(parents=True, exist_ok=True); output_file.write_text(compressed.text)
        except (IOError, OSError, PermissionError) as exc: typer.secho(f"Error writing {output_file}: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
        typer.echo(f"Saved compressed output to {output_file}")
    else: typer.echo(compressed.text)
    if trace_file and trace:
        try: trace_file.parent.mkdir(parents=True, exist_ok=True); trace_file.write_text(json.dumps(asdict(trace)))
        except (IOError, OSError, PermissionError) as exc: typer.secho(f"Error writing {trace_file}: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

def _process_file_compression(file_path: Path, strategy_id: str, budget: int, output_file: Optional[Path], verbose_stats: bool, tokenizer: Any, trace_file: Optional[Path]) -> None:
    if not file_path.exists() or not file_path.is_file(): typer.secho(f"File not found: {file_path}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    try: text_content = file_path.read_text()
    except (IOError, OSError, PermissionError) as exc: typer.secho(f"Error reading {file_path}: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    _process_string_compression(text_content, strategy_id, budget, output_file, trace_file, verbose_stats, tokenizer)

def _process_directory_compression(dir_path: Path, strategy_id: str, budget: int, output_dir_param: Optional[Path], recursive: bool, pattern: str, verbose_stats: bool, tokenizer: Any, trace_file: Optional[Path]) -> None:
    if not dir_path.exists() or not dir_path.is_dir(): typer.secho(f"Directory not found: {dir_path}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    files = list(dir_path.rglob(pattern) if recursive else dir_path.glob(pattern))
    if not files: typer.echo("No matching files found."); return
    if trace_file is not None: typer.secho("--output-trace ignored for directory input", fg=typer.colors.YELLOW); trace_file = None
    count = 0
    for input_file in files:
        typer.echo(f"Processing {input_file}...")
        if output_dir_param:
            try: output_dir_param.mkdir(parents=True, exist_ok=True)
            except (IOError, OSError, PermissionError) as exc: typer.secho(f"Error creating {output_dir_param}: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
            rel = input_file.relative_to(dir_path); out_path = output_dir_param / rel; out_path.parent.mkdir(parents=True, exist_ok=True)
        else: out_path = input_file.with_name(f"{input_file.stem}_compressed{input_file.suffix}")
        _process_file_compression(input_file, strategy_id, budget, out_path, verbose_stats, tokenizer, None); count += 1
    if verbose_stats: typer.echo(f"Processed {count} files.")

# --- Dev App Commands ---
@dev_app.command("list-metrics")
def list_metrics() -> None:
    for mid in sorted(_VALIDATION_METRIC_REGISTRY): typer.echo(mid)

@dev_app.command("list-strategies")
def list_strategies() -> None:
    load_plugins()
    table = Table("Strategy ID", "Display Name", "Version", "Source", "Status")
    meta = all_strategy_metadata()
    for sid in available_strategies():
        info = meta.get(sid, {}); status = "";
        if info.get("overrides"): status = f"Overrides {info['overrides']}"
        table.add_row(sid, info.get("display_name", sid) or sid, info.get("version", "N/A") or "N/A", info.get("source", "built-in") or "built-in", status)
    console.print(table)

@dev_app.command("inspect-strategy")
def inspect_strategy(strategy_name: str, *, ctx: typer.Context, memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Memory store directory (if different from global --memory-path)"), list_prototypes: bool = typer.Option(False, "--list-prototypes", help="List prototypes for the strategy")) -> None:
    if strategy_name != "prototype": typer.echo(f"Unknown strategy: {strategy_name}", err=True); raise typer.Exit(code=1)
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    agent = _load_agent(path)
    if list_prototypes:
        protos = agent.get_prototypes_view()
        table = Table("id", "strength", "confidence", "summary", title="Beliefs")
        for p in protos: table.add_row(p["id"], f"{p['strength']:.2f}", f"{p['confidence']:.2f}", p["summary"][:60])
        console.print(table)

@dev_app.command("evaluate-compression")
def evaluate_compression_cmd(original_input: str = typer.Argument(..., help="Original text or file path"), compressed_input: str = typer.Argument(..., help="Compressed text or file path, or '-' for stdin"), metric_id: str = typer.Option(..., "--metric", "-m", help="Metric ID"), metric_params_json: Optional[str] = typer.Option(None, "--metric-params", help="Metric parameters as JSON"), json_output: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    def read_input(value: str, allow_stdin: bool) -> str:
        if value == "-":
            if not allow_stdin: typer.secho("stdin already used", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(value);
        if p.exists() and p.is_file(): return p.read_text()
        return value
    orig_text = read_input(original_input, True); comp_text = read_input(compressed_input, compressed_input == "-")
    try: Metric = get_validation_metric_class(metric_id)
    except KeyError: typer.secho(f"Unknown metric '{metric_id}'", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    params = {}
    if metric_params_json:
        try: params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc: typer.secho(f"Invalid metric params: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    metric = Metric(**params)
    try: scores = metric.evaluate(original_text=orig_text, compressed_text=comp_text)
    except Exception as exc: typer.secho(f"Metric error: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    if json_output: typer.echo(json.dumps(scores))
    else:
        for k, v in scores.items(): typer.echo(f"{k}: {v}")

@dev_app.command("test-llm-prompt")
def test_llm_prompt(*, context_input: str = typer.Option(..., "--context", "-c", help="Compressed context string, path, or '-' for stdin"), query: str = typer.Option(..., "--query", "-q", help="User query"), model_id: str = typer.Option("tiny-gpt2", "--model", help="Model ID"), system_prompt: Optional[str] = typer.Option(None, "--system-prompt", "-s", help="Optional system prompt"), max_new_tokens: int = typer.Option(150, help="Max new tokens"), output_llm_response_file: Optional[Path] = typer.Option(None, "--output-response", help="File to save response"), llm_config_file: Optional[Path] = typer.Option(Path("llm_models_config.yaml"), "--llm-config", exists=True, dir_okay=False, resolve_path=True, help="LLM config file"), api_key_env_var: Optional[str] = typer.Option(None, help="Env var for API key")) -> None:
    def read_val(val: str) -> str:
        if val == "-": return sys.stdin.read()
        p = Path(val);
        if p.exists() and p.is_file(): return p.read_text()
        return val
    context_text = read_val(context_input); cfg = {}
    if llm_config_file:
        try: cfg = yaml.safe_load(llm_config_file.read_text()) or {}
        except Exception as exc: typer.secho(f"Error loading config: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    model_cfg = cfg.get(model_id, {"provider": "local", "model_name": model_id}); provider_name = model_cfg.get("provider"); model_name = model_cfg.get("model_name", model_id)
    if provider_name == "openai": provider = llm_providers.OpenAIProvider()
    elif provider_name == "gemini": provider = llm_providers.GeminiProvider()
    else: provider = llm_providers.LocalTransformersProvider()
    api_key = os.getenv(api_key_env_var) if api_key_env_var else None; prompt_parts = []
    if system_prompt: prompt_parts.append(system_prompt)
    prompt_parts.append(context_text); prompt_parts.append(query); prompt = "\n".join(part for part in prompt_parts if part)
    try: response = provider.generate_response(prompt, model_name=model_name, max_new_tokens=max_new_tokens, api_key=api_key)
    except Exception as exc: typer.secho(f"LLM error: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    if output_llm_response_file:
        try: output_llm_response_file.write_text(response)
        except Exception as exc: typer.secho(f"Error writing {output_llm_response_file}: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    else: typer.echo(response)

@dev_app.command("evaluate-llm-response")
def evaluate_llm_response_cmd(response_input: str = typer.Argument(..., help="LLM response text or file, or '-' for stdin"), reference_input: str = typer.Argument(..., help="Reference answer text or file"), metric_id: str = typer.Option(..., "--metric", "-m", help="Metric ID"), metric_params_json: Optional[str] = typer.Option(None, "--metric-params", help="Metric parameters as JSON"), json_output: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    def read_value(val: str, allow_stdin: bool) -> str:
        if val == "-":
            if not allow_stdin: typer.secho("stdin already used", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(val);
        if p.exists() and p.is_file(): return p.read_text()
        return val
    resp_text = read_value(response_input, True); ref_text = read_value(reference_input, False)
    try: Metric = get_validation_metric_class(metric_id)
    except KeyError: typer.secho(f"Unknown metric '{metric_id}'", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    params = {}
    if metric_params_json:
        try: params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc: typer.secho(f"Invalid metric params: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    metric = Metric(**params)
    try: scores = metric.evaluate(llm_response=resp_text, reference_answer=ref_text)
    except Exception as exc: typer.secho(f"Metric error: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    if json_output: typer.echo(json.dumps(scores))
    else:
        for k, v in scores.items(): typer.echo(f"{k}: {v}")

@dev_app.command("download-embedding-model")
def download_embedding_model_cli(model_name: str = typer.Option("all-MiniLM-L6-v2", help="SentenceTransformer model name")) -> None:
    bar = tqdm(total=1, desc="Downloading embedding model", disable=False)
    util_download_embedding_model(model_name)
    bar.update(1); bar.close(); typer.echo(f"Downloaded {model_name}")

@dev_app.command("download-chat-model")
def download_chat_model_cli(model_name: str = typer.Option("tiny-gpt2", help="Local causal LM name")) -> None:
    bar = tqdm(total=1, desc="Downloading chat model", disable=False)
    util_download_chat_model(model_name)
    bar.update(1); bar.close(); typer.echo(f"Downloaded {model_name}")

@dev_app.command("create-strategy-package")
def create_strategy_package(name: str = typer.Option("sample_strategy", "--name", help="Strategy name"), path: Optional[Path] = typer.Option(None, "--path", help="Output directory")) -> None:
    target = Path(path or name); target.mkdir(parents=True, exist_ok=True)
    (target / "experiments").mkdir(exist_ok=True)
    (target / "strategy.py").write_text(f"""from gist_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace\n\n\nclass MyStrategy(CompressionStrategy):\n    id = "{name}"\n\n    def compress(self, text_or_chunks, llm_token_budget, **kwargs):\n        return CompressedMemory(text=str(text_or_chunks)), CompressionTrace()\n""")
    manifest = {"package_format_version": "1.0", "strategy_id": name, "strategy_class_name": "MyStrategy", "strategy_module": "strategy", "display_name": name, "version": "0.1.0", "authors": [], "description": "Describe the strategy"}
    (target / "strategy_package.yaml").write_text(yaml.safe_dump(manifest))
    (target / "requirements.txt").write_text("\n"); (target / "README.md").write_text(f"# {name}\n")
    (target / "experiments" / "example.yaml").write_text("""dataset: example.txt\nparam_grid:\n- {}\npackaged_strategy_config:\n  strategy_params: {}\n""")
    typer.echo(f"Created package at {target}")

@dev_app.command("validate-strategy-package")
def validate_strategy_package(package_path: Path) -> None:
    errors, warnings = validate_package_dir(package_path)
    for w in warnings: typer.secho(f"Warning: {w}", fg=typer.colors.YELLOW)
    if errors:
        for e in errors: typer.echo(e)
        raise typer.Exit(code=1)
    typer.echo("Package is valid")

@dev_app.command("run-package-experiment")
def run_package_experiment(package_path: Path = typer.Argument(..., help="Path to strategy package"), experiment: Optional[str] = typer.Option(None, "--experiment", help="Experiment config")) -> None:
    manifest = load_manifest(package_path / "strategy_package.yaml"); Strategy = load_strategy_class(package_path, manifest)
    missing = check_requirements_installed(package_path / "requirements.txt")
    if missing: typer.secho(f"Warning: missing requirements - {', '.join(missing)}", fg=typer.colors.YELLOW)
    if experiment is None:
        defaults = manifest.get("default_experiments", [])
        if len(defaults) == 1: experiment = defaults[0].get("path")
    if experiment is None: typer.secho("No experiment specified", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    exp_path = Path(experiment)
    if not exp_path.is_absolute(): exp_path = package_path / exp_path
    cfg_data = yaml.safe_load(exp_path.read_text()) or {}; dataset = cfg_data.get("dataset")
    if dataset and not Path(dataset).is_absolute(): cfg_data["dataset"] = str((package_path / dataset).resolve())
    params = cfg_data.get("param_grid", [{}])
    cfg = ResponseExperimentConfig(dataset=Path(cfg_data["dataset"]), param_grid=params, validation_metrics=cfg_data.get("validation_metrics"))
    strategy_params = cfg_data.get("packaged_strategy_config", {}).get("strategy_params", {}); strategy = Strategy(**strategy_params)
    results = run_response_experiment(cfg, strategy=strategy); typer.echo(json.dumps(results))

@dev_app.command("run-hpo-script")
def run_hpo_script(script: Path = typer.Argument(..., help="Python script")) -> None:
    if not script.exists(): typer.secho(f"Script '{script}' not found", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    runpy.run_path(str(script), run_name="__main__")

@dev_app.command("inspect-trace")
def inspect_trace(trace_file: Path = typer.Argument(..., help="Path to trace JSON"), step_type: Optional[str] = typer.Option(None, "--type", help="Filter by step type")) -> None:
    if not trace_file.exists(): typer.secho(f"Trace file '{trace_file}' not found", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    data = json.loads(trace_file.read_text()); steps = data.get("steps", [])
    console.print(f"Strategy: {data.get('strategy_name', '')}")
    table = Table("idx", "type", "details")
    for idx, step in enumerate(steps):
        stype = step.get("type")
        if step_type and stype != step_type: continue
        preview = json.dumps(step.get("details", {}))[:50]; table.add_row(str(idx), stype or "", preview)
    console.print(table)

# Commands still under @app.command that need to be moved or have their functionality subsumed
# @app.command()
# def validate_cmd(...) # This should be agent_app.command("validate")
# @app.command("download-model") -> now dev_app.command("download-embedding-model")
# @app.command("download-chat-model") -> now dev_app.command("download-chat-model")

if __name__ == "__main__":
	app()
