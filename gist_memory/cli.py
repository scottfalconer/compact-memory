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
    help="Gist Memory: A CLI for intelligent information management using memory agents and advanced compression. Ingest, query, and summarize information. Manage agent configurations and developer tools."
)
console = Console()

# --- New Command Groups ---
agent_app = typer.Typer(help="Manage memory agents: initialize, inspect statistics, validate, and clear.")
config_app = typer.Typer(help="Manage Gist Memory application configuration settings.")
dev_app = typer.Typer(help="Developer tools for testing, evaluation, and managing extension packages.")

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
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Path to write debug logs. If not set, logs are not written to file."),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Enable verbose (DEBUG level) logging to console and log file (if specified)."),
    memory_path: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Path to the Gist Memory agent directory. Overrides GIST_MEMORY_PATH env var and configuration files."),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Default model ID for LLM interactions. Overrides GIST_MEMORY_DEFAULT_MODEL_ID env var and configuration files."),
    strategy_id: Optional[str] = typer.Option(None, "--strategy-id", help="Default compression strategy ID. Overrides GIST_MEMORY_DEFAULT_STRATEGY_ID env var and configuration files."),
    version: Optional[bool] = typer.Option(None, "-v", "--version", callback=version_callback, is_eager=True, help="Show the application version and exit."),
) -> None:
    # Ensure config is loaded early, before other logic might depend on it.
    # It's important that ctx.obj['config'] is populated.
    if ctx.obj is None: ctx.obj = {}
    if 'config' not in ctx.obj: # Could be pre-populated by a test harness or similar
        ctx.obj['config'] = Config()

    config: Config = ctx.obj['config']
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

    # ctx.obj['config'] is already set above
    ctx.obj.update({
        "verbose": resolved_verbose, "log_file": resolved_log_file,
        "gist_memory_path": resolved_memory_path, "default_model_id": resolved_model_id,
        "default_strategy_id": resolved_strategy_id,
        # "config": config, # Already present
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
@agent_app.command("init", help="Initializes a new Gist Memory agent in a specified directory.\n\nUsage Examples:\n  gist-memory agent init ./my_agent_dir\n  gist-memory agent init /path/to/another_agent --name 'research_agent' --model-name 'sentence-transformers/all-mpnet-base-v2' --tau 0.75")
def init(
    target_directory: Path = typer.Argument(..., help="Directory to initialize the new agent in. Will be created if it doesn't exist.", resolve_path=True),
    *,
    ctx: typer.Context,
    name: str = typer.Option("default", help="A descriptive name for the agent."),
    model_name: str = typer.Option("all-MiniLM-L6-v2", help="Name of the sentence-transformer model for embeddings."),
    tau: float = typer.Option(0.8, help="Similarity threshold (tau) for memory consolidation, between 0.5 and 0.95."), # Added range to help
    alpha: float = typer.Option(0.1, help="Alpha parameter, controlling the decay rate for memory importance."),
    chunker: str = typer.Option("sentence_window", help="Chunking strategy to use for processing text during ingestion.")
) -> None:
    path = target_directory.expanduser()
    if path.exists() and any(path.iterdir()): typer.secho(f"Error: Directory '{path}' already exists and is not empty.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    if not 0.5 <= tau <= 0.95: typer.secho("Error: --tau must be between 0.5 and 0.95.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1) # Keep runtime check
    try: dim = get_embedding_dim()
    except RuntimeError as exc: typer.echo(str(exc), err=True); raise typer.Exit(code=1)
    store = JsonNpyVectorStore(path=str(path), embedding_model=model_name, embedding_dim=dim)
    store.meta.update({"agent_name": name, "tau": tau, "alpha": alpha, "chunker": chunker})
    store.save(); typer.echo(f"Successfully initialized Gist Memory agent at {path}")

@agent_app.command("stats", help="Displays statistics about the Gist Memory agent.\n\nUsage Examples:\n  gist-memory agent stats\n  gist-memory agent stats --memory-path path/to/my_agent --json")
def stats(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Path to the agent directory. Overrides global setting if provided."),
    json_output: bool = typer.Option(False, "--json", help="Output statistics in JSON format.")
) -> None:
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for stats.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    agent = _load_agent(path); data = agent.get_statistics(); logging.debug("Collected statistics: %s", data)
    if json_output: typer.echo(json.dumps(data))
    else:
        for k, v in data.items(): typer.echo(f"{k}: {v}")

@agent_app.command("validate", help="Validates the integrity of the agent's storage.")
def validate_agent_storage(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Path to the agent directory. Overrides global setting if provided.")
) -> None:
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for validate.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    try: JsonNpyVectorStore(path=str(path))
    except EmbeddingDimensionMismatchError as exc: typer.secho(f"Embedding dimension mismatch: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    except Exception as exc: typer.secho(f"Error loading agent: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    typer.echo("Agent storage is valid.")

@agent_app.command("clear", help="Deletes all data from an agent's memory. This action is irreversible.\n\nUsage Examples:\n  gist-memory agent clear --force\n  gist-memory agent clear --memory-path path/to/another_agent --dry-run")
def clear(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Path to the agent directory. Overrides global setting if provided."),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without prompting for confirmation."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate deletion and show what would be deleted without actually removing files.")
) -> None:
    final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for clear.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    if dry_run: store = JsonNpyVectorStore(path=str(path)); typer.echo(f"Dry run: Would delete {len(store.prototypes)} prototypes and {len(store.memories)} memories from agent at '{path}'."); return # Added path to message
    if not force:
        if not typer.confirm(f"Are you sure you want to delete all data in agent at '{path}'? This cannot be undone.", abort=True): return # Added path and warning
    if path.exists(): shutil.rmtree(path); typer.echo(f"Successfully cleared agent data at {path}")
    else: typer.secho(f"Error: Directory '{path}' not found.", err=True, fg=typer.colors.RED) # Added path

# --- Top-Level Commands ---
@app.command("ingest", help="Ingests text from a file or directory into the agent's memory.\n\nUsage Examples:\n  gist-memory ingest path/to/my_data.txt --tau 0.7\n  gist-memory ingest path/to/my_directory/")
def ingest(
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="Path to the text file or directory containing text files to ingest.", exists=True, file_okay=True, dir_okay=True, resolve_path=True),
    tau: Optional[float] = typer.Option(None, "--tau", "-t", help="Similarity threshold (0.5-0.95) for memory consolidation. Overrides agent's existing tau if set. If agent is new, this tau is used for initialization."),
    json_output: bool = typer.Option(False, "--json", help="Output ingestion summary statistics in JSON format.")
) -> None:
    """
    Ingests text from a specified file or all text files in a directory into the agent's memory.
    The agent is determined by the global --memory-path option or configuration settings.
    If the agent does not exist, it will be initialized with the provided --tau (or default).
    """
    memory_path_str = ctx.obj.get("gist_memory_path")
    if not memory_path_str:
        typer.secho("Error: Gist Memory path not set. Use --memory-path option, GIST_MEMORY_PATH env var, or set in config.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    agent_path = Path(memory_path_str) # Ensured this is a Path object for consistency
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
        dataset=source,
        similarity_threshold=final_tau,
        work_dir=agent_path
    )

    typer.echo(f"Ingesting data from '{source}' into agent at '{agent_path}' with tau={final_tau}...")
    try:
        metrics = run_experiment(cfg) # run_experiment handles agent loading/initialization
        if json_output:
            typer.echo(json.dumps(metrics))
        else:
            for k, v_val in metrics.items():
                typer.echo(f"{k}: {v_val}")
        typer.echo(f"Ingestion from '{source}' complete.")
    except Exception as e:
        typer.secho(f"Error during ingestion from '{source}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@app.command("query", help="Queries the Gist Memory agent and returns an AI-generated response.\n\nUsage Examples:\n  gist-memory query \"What is the capital of France?\"\n  gist-memory query \"Explain the theory of relativity in simple terms\" --show-prompt-tokens")
def query(
    ctx: typer.Context,
    query_text: str = typer.Argument(..., help="The query text to send to the agent."),
    show_prompt_tokens: bool = typer.Option(False, "--show-prompt-tokens", help="Display the token count of the final prompt sent to the LLM.")
) -> None:
    final_memory_path_str = ctx.obj.get("gist_memory_path")
    if final_memory_path_str is None: typer.secho("Critical Error: Memory path could not be resolved for query.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1) # Should be caught by main usually
    path = Path(final_memory_path_str)
    if not path.exists(): typer.secho(f"Error: Gist Memory path '{path}' not found or is invalid. Initialize an agent first using 'agent init'.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    with PersistenceLock(path):
        agent = _load_agent(path)
        final_model_id = ctx.obj.get("default_model_id") # Renamed for clarity
        final_strategy_id = ctx.obj.get("default_strategy_id") # Renamed for clarity

        if final_model_id is None: typer.secho("Error: Default Model ID not specified. Use --model-id option or set in config.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)

        try:
            agent._chat_model = local_llm.LocalChatModel(model_name=final_model_id)
            agent._chat_model.load_model()
        except RuntimeError as exc: typer.secho(f"Error loading chat model '{final_model_id}': {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1) # Added model_id to error

        mgr = ActiveMemoryManager()
        comp_strategy_instance = None # Renamed for clarity
        if final_strategy_id and final_strategy_id.lower() != "none":
            try:
                comp_cls = get_compression_strategy(final_strategy_id)
                comp_strategy_instance = comp_cls()
            except KeyError: typer.secho(f"Error: Unknown compression strategy '{final_strategy_id}' (from global config/option). Available: {', '.join(available_strategies())}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1) # Added available strategies

        try:
            result = agent.receive_channel_message("cli", query_text, mgr, compression=comp_strategy_instance)
        except TypeError: # Older agents might not have compression param
            result = agent.receive_channel_message("cli", query_text, mgr)

        reply = result.get("reply")
        if reply: typer.echo(reply)
        else: typer.secho("The agent did not return a reply.", fg=typer.colors.YELLOW)

        if show_prompt_tokens and result.get("prompt_tokens") is not None:
            typer.echo(f"Prompt tokens: {result['prompt_tokens']}")

@app.command("summarize", help="Compresses text content from a string, file, or directory using a specified strategy and budget.\n\nUsage Examples:\n  gist-memory summarize \"Some very long text...\" --strategy first_last --budget 100\n  gist-memory summarize path/to/document.txt -s prototype -b 200 -o summary.txt\n  gist-memory summarize input_dir/ -s custom_package_strat -b 500 -o output_dir/ --recursive -p \"*.md\"")
def summarize(
    ctx: typer.Context,
    input_source: str = typer.Argument(..., help="Input text directly, or a path to a text file or directory of text files to summarize."),
    *,
    strategy_arg: Optional[str] = typer.Option(None, "--strategy", "-s", help="Compression strategy ID to use. Overrides the global default strategy."),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", resolve_path=True, help="File path to write compressed output. For directory input, this is the root output directory. If unspecified, prints to console."),
    output_trace: Optional[Path] = typer.Option(None, "--output-trace", resolve_path=True, help="File path to write the CompressionTrace JSON object. (Not applicable for directory input)."),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Process text files in subdirectories recursively when 'input_source' is a directory."),
    pattern: str = typer.Option("*.txt", "-p", "--pattern", help="File glob pattern to match files when 'input_source' is a directory (e.g., '*.md', '**/*.txt')."),
    budget: int = typer.Option(..., help="Token budget for the compressed output. The strategy will aim to keep the output within this limit."),
    verbose_stats: bool = typer.Option(False, "--verbose-stats", help="Show detailed token counts and processing time per item.")
) -> None:
    final_strategy_id = strategy_arg if strategy_arg is not None else ctx.obj.get("default_strategy_id")
    if not final_strategy_id: typer.secho("Error: Compression strategy not specified. Use --strategy option or set GIST_MEMORY_DEFAULT_STRATEGY_ID / config.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)
    try: import tiktoken; enc = tiktoken.get_encoding("gpt2"); tokenizer = lambda text, **k: {"input_ids": enc.encode(text)} # TODO: make tokenizer configurable
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
@dev_app.command("list-metrics", help="Lists all available validation metric IDs that can be used in evaluations.")
def list_metrics() -> None:
    """Lists all registered validation metric IDs."""
    if not _VALIDATION_METRIC_REGISTRY:
        typer.echo("No validation metrics found.")
        return
    typer.echo("Available validation metric IDs:")
    for mid in sorted(_VALIDATION_METRIC_REGISTRY):
        typer.echo(f"- {mid}")

@dev_app.command("list-strategies", help="Lists all available compression strategy IDs, their versions, and sources (built-in or plugin).")
def list_strategies() -> None:
    """Displays a table of all registered compression strategies, including plugins."""
    load_plugins() # Ensure plugins are loaded
    table = Table("Strategy ID", "Display Name", "Version", "Source", "Status", title="Available Compression Strategies")
    meta = all_strategy_metadata()
    ids = available_strategies()
    if not ids:
        typer.echo("No compression strategies found.")
        return
    for sid in sorted(ids):
        info = meta.get(sid, {})
        status = ""
        if info.get("overrides"):
            status = f"Overrides '{info['overrides']}'"
        table.add_row(
            sid,
            info.get("display_name", sid) or sid,
            info.get("version", "N/A") or "N/A",
            info.get("source", "built-in") or "built-in",
            status
        )
    console.print(table)

@dev_app.command("inspect-strategy", help="Inspects aspects of a compression strategy, currently focused on 'prototype' strategy's beliefs.")
def inspect_strategy(
    strategy_name: str = typer.Argument(..., help="The name of the strategy to inspect. Currently, only 'prototype' is supported."),
    *,
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Path to the agent directory. Overrides global setting if provided. Required if 'list-prototypes' is used."),
    list_prototypes: bool = typer.Option(False, "--list-prototypes", help="List consolidated prototypes (beliefs) if the strategy is 'prototype' and an agent path is provided.")
) -> None:
    if strategy_name.lower() != "prototype":
        typer.secho(f"Error: Inspection for strategy '{strategy_name}' is not supported. Only 'prototype' is currently inspectable.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    if list_prototypes:
        final_memory_path_str = memory_path_arg if memory_path_arg is not None else ctx.obj.get("gist_memory_path")
        if not final_memory_path_str:
            typer.secho("Error: --memory-path is required when --list-prototypes is used.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)

        path = Path(final_memory_path_str)
        if not path.exists(): typer.secho(f"Error: Gist Memory path '{path}' not found or is invalid.", fg=typer.colors.RED, err=True); raise typer.Exit(code=1)

        agent = _load_agent(path)
        protos = agent.get_prototypes_view()
        if not protos:
            typer.echo(f"No prototypes found in agent at '{path}'.")
            return
        table = Table("ID", "Strength", "Confidence", "Summary", title=f"Prototypes for Agent at '{path}'")
        for p in protos: table.add_row(p["id"], f"{p['strength']:.2f}", f"{p['confidence']:.2f}", p["summary"][:80]) # Increased summary length
        console.print(table)
    else:
        typer.echo(f"Strategy '{strategy_name}' is available. Use --list-prototypes and provide an agent path to see its beliefs.")


@dev_app.command("evaluate-compression", help="Evaluates compressed text against original text using a specified metric.\n\nUsage Examples:\n  gist-memory dev evaluate-compression original.txt summary.txt --metric compression_ratio\n  echo \"original text\" | gist-memory dev evaluate-compression - summary.txt --metric some_other_metric --metric-params '{\"param\": \"value\"}'")
def evaluate_compression_cmd(
    original_input: str = typer.Argument(..., help="Original text content, path to a text file, or '-' to read from stdin."),
    compressed_input: str = typer.Argument(..., help="Compressed text content, path to a text file, or '-' to read from stdin."),
    metric_id: str = typer.Option(..., "--metric", "-m", help="ID of the validation metric to use (see 'list-metrics')."),
    metric_params_json: Optional[str] = typer.Option(None, "--metric-params", help="Metric parameters as a JSON string (e.g., '{\"model_name\": \"bert-base-uncased\"}')."),
    json_output: bool = typer.Option(False, "--json", help="Output evaluation scores in JSON format.")
) -> None:
    def read_input(value: str, allow_stdin: bool) -> str:
        if value == "-":
            if not allow_stdin: typer.secho("Error: Cannot use '-' for stdin for both original and compressed input simultaneously.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(value)
        if p.exists() and p.is_file():
            try: return p.read_text()
            except Exception as e: typer.secho(f"Error reading file '{p}': {e}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
        return value # Treat as direct text content if not a file or '-'

    orig_text = read_input(original_input, True)
    comp_text = read_input(compressed_input, original_input != "-") # Only allow stdin for compressed if original isn't stdin

    try: Metric = get_validation_metric_class(metric_id)
    except KeyError: typer.secho(f"Error: Unknown metric ID '{metric_id}'. Use 'list-metrics' to see available IDs.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    params = {}
    if metric_params_json:
        try: params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc: typer.secho(f"Error: Invalid JSON in --metric-params: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    metric = Metric(**params)
    try: scores = metric.evaluate(original_text=orig_text, compressed_text=comp_text)
    except Exception as exc: typer.secho(f"Error during metric evaluation: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    if json_output: typer.echo(json.dumps(scores))
    else:
        typer.echo(f"Scores for metric '{metric_id}':")
        for k, v in scores.items(): typer.echo(f"- {k}: {v}")

@dev_app.command("test-llm-prompt", help="Tests a Language Model (LLM) prompt with specified context and query.\n\nUsage Examples:\n  gist-memory dev test-llm-prompt --context \"AI is rapidly evolving.\" --query \"Tell me more.\" --model-id tiny-gpt2\n  cat context.txt | gist-memory dev test-llm-prompt --context - -q \"What are the implications?\" --model-id openai/gpt-3.5-turbo --output-response response.txt --llm-config my_llm_config.yaml")
def test_llm_prompt(
    *,
    context_input: str = typer.Option(..., "--context", "-c", help="Context string for the LLM, path to a context file, or '-' to read from stdin."),
    query: str = typer.Option(..., "--query", "-q", help="User query to append to the context for the LLM."),
    model_id: str = typer.Option("tiny-gpt2", "--model", help="Model ID to use for the test (must be defined in LLM config)."),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", "-s", help="Optional system prompt to prepend to the main prompt."),
    max_new_tokens: int = typer.Option(150, help="Maximum number of new tokens the LLM should generate."),
    output_llm_response_file: Optional[Path] = typer.Option(None, "--output-response", help="File path to save the LLM's raw response. If unspecified, prints to console."),
    llm_config_file: Optional[Path] = typer.Option(Path("llm_models_config.yaml"), "--llm-config", exists=True, dir_okay=False, resolve_path=True, help="Path to the LLM configuration YAML file."),
    api_key_env_var: Optional[str] = typer.Option(None, help="Environment variable name that holds the API key for the LLM provider (e.g., 'OPENAI_API_KEY').")
) -> None:
    def read_val(val: str, input_name: str) -> str: # Added input_name for better error messages
        if val == "-": return sys.stdin.read()
        p = Path(val)
        if p.exists() and p.is_file():
            try: return p.read_text()
            except Exception as e: typer.secho(f"Error reading {input_name} file '{p}': {e}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
        return val # Treat as direct content

    context_text = read_val(context_input, "context")

    cfg = {}
    if llm_config_file and llm_config_file.exists(): # Check existence again as Typer's check might be bypassed in some complex setups
        try: cfg = yaml.safe_load(llm_config_file.read_text()) or {}
        except Exception as exc: typer.secho(f"Error loading LLM config '{llm_config_file}': {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    model_cfg = cfg.get(model_id, {"provider": "local", "model_name": model_id})
    provider_name = model_cfg.get("provider")
    actual_model_name = model_cfg.get("model_name", model_id) # Use actual_model_name to avoid conflict

    if provider_name == "openai": provider = llm_providers.OpenAIProvider()
    elif provider_name == "gemini": provider = llm_providers.GeminiProvider()
    else: provider = llm_providers.LocalTransformersProvider() # Default or if provider_name is 'local'

    api_key = os.getenv(api_key_env_var) if api_key_env_var else None

    prompt_parts = []
    if system_prompt: prompt_parts.append(system_prompt)
    if context_text: prompt_parts.append(context_text) # Ensure context_text is not empty before adding
    prompt_parts.append(query)
    prompt = "\n\n".join(part for part in prompt_parts if part) # Use double newline for better separation

    typer.echo(f"--- Sending Prompt to LLM ({provider_name} - {actual_model_name}) ---")
    typer.echo(prompt[:500] + "..." if len(prompt) > 500 else prompt) # Preview long prompts
    typer.echo("--- End of Prompt ---")

    try:
        response = provider.generate_response(prompt, model_name=actual_model_name, max_new_tokens=max_new_tokens, api_key=api_key)
    except Exception as exc: typer.secho(f"LLM generation error: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    if output_llm_response_file:
        try:
            output_llm_response_file.parent.mkdir(parents=True, exist_ok=True)
            output_llm_response_file.write_text(response)
            typer.echo(f"LLM response saved to: {output_llm_response_file}")
        except Exception as exc: typer.secho(f"Error writing LLM response to '{output_llm_response_file}': {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    else:
        typer.echo("\n--- LLM Response ---")
        typer.echo(response)

@dev_app.command("evaluate-llm-response", help="Evaluates an LLM's response against a reference answer using a specified metric.")
def evaluate_llm_response_cmd(
    response_input: str = typer.Argument(..., help="LLM's generated response text, path to a response file, or '-' to read from stdin."),
    reference_input: str = typer.Argument(..., help="Reference (ground truth) answer text or path to a reference file."),
    metric_id: str = typer.Option(..., "--metric", "-m", help="ID of the validation metric to use (see 'list-metrics')."),
    metric_params_json: Optional[str] = typer.Option(None, "--metric-params", help="Metric parameters as a JSON string (e.g., '{\"model_name\": \"bert-base-uncased\"}')."),
    json_output: bool = typer.Option(False, "--json", help="Output evaluation scores in JSON format.")
) -> None:
    def read_value(val: str, input_name: str, allow_stdin: bool) -> str: # Added input_name
        if val == "-":
            if not allow_stdin: typer.secho(f"Error: Cannot use '-' for stdin for both response and reference input simultaneously.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(val)
        if p.exists() and p.is_file():
            try: return p.read_text()
            except Exception as e: typer.secho(f"Error reading {input_name} file '{p}': {e}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
        return val # Treat as direct content

    resp_text = read_value(response_input, "LLM response", True)
    ref_text = read_value(reference_input, "reference answer", response_input != "-")

    try: Metric = get_validation_metric_class(metric_id)
    except KeyError: typer.secho(f"Error: Unknown metric ID '{metric_id}'. Use 'list-metrics' to see available IDs.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    params = {}
    if metric_params_json:
        try: params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc: typer.secho(f"Error: Invalid JSON in --metric-params: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    metric = Metric(**params)
    try: scores = metric.evaluate(llm_response=resp_text, reference_answer=ref_text)
    except Exception as exc: typer.secho(f"Error during metric evaluation: {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    if json_output: typer.echo(json.dumps(scores))
    else:
        typer.echo(f"Scores for metric '{metric_id}':")
        for k, v in scores.items(): typer.echo(f"- {k}: {v}")

@dev_app.command("download-embedding-model", help="Downloads a specified SentenceTransformer embedding model from Hugging Face.")
def download_embedding_model_cli(model_name: str = typer.Option("all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to download (e.g., 'all-MiniLM-L6-v2').")) -> None:
    typer.echo(f"Starting download for embedding model: {model_name}...")
    bar = tqdm(total=1, desc=f"Downloading {model_name}", unit="model", disable=False)
    try:
        util_download_embedding_model(model_name)
        bar.update(1)
        typer.echo(f"Successfully downloaded embedding model '{model_name}'.")
    except Exception as e:
        typer.secho(f"Error downloading embedding model '{model_name}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    finally:
        bar.close()

@dev_app.command("download-chat-model", help="Downloads a specified causal Language Model (e.g., for chat) from Hugging Face.")
def download_chat_model_cli(model_name: str = typer.Option("tiny-gpt2", help="Name of the Hugging Face causal LM to download (e.g., 'gpt2', 'facebook/opt-125m').")) -> None:
    typer.echo(f"Starting download for chat model: {model_name}...")
    bar = tqdm(total=1, desc=f"Downloading {model_name}", unit="model", disable=False)
    try:
        util_download_chat_model(model_name)
        bar.update(1)
        typer.echo(f"Successfully downloaded chat model '{model_name}'.")
    except Exception as e:
        typer.secho(f"Error downloading chat model '{model_name}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    finally:
        bar.close()

@dev_app.command("create-strategy-package", help="Creates a new compression strategy extension package from a template.")
def create_strategy_package(
    name: str = typer.Option("sample_strategy", "--name", help="Name for the new strategy (e.g., 'my_custom_strategy'). Used for directory and strategy ID."),
    path: Optional[Path] = typer.Option(None, "--path", help="Directory where the strategy package will be created. Defaults to a new directory named after the strategy in the current location.")
) -> None:
    target_dir = Path(path or name).resolve() # Renamed for clarity

    if target_dir.exists() and any(target_dir.iterdir()):
        typer.secho(f"Error: Output directory '{target_dir}' already exists and is not empty.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "experiments").mkdir(exist_ok=True)

    strategy_py_content = f"""from gist_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace
# Add any other necessary imports here

class MyStrategy(CompressionStrategy):
    # Unique identifier for your strategy
    id = "{name}"

    # Optional: Define parameters your strategy accepts with default values
    # def __init__(self, param1: int = 10, param2: str = "default"):
    #     self.param1 = param1
    #     self.param2 = param2

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        '''
        Compresses the input text or chunks to meet the token budget.

        Args:
            text_or_chunks: Either a single string or a list of strings (chunks).
            llm_token_budget: The maximum number of tokens the compressed output should ideally have.
            **kwargs: Additional keyword arguments, often including 'tokenizer'.

        Returns:
            A tuple containing:
                - CompressedMemory: An object with the 'text' attribute holding the compressed string.
                - CompressionTrace: An object detailing the steps and outcomes of the compression.
        '''
        tokenizer = kwargs.get("tokenizer") # Example of getting tokenizer if needed

        # --- Your compression logic here ---
        # This is a placeholder. Implement your actual compression algorithm.
        compressed_text = str(text_or_chunks)[:llm_token_budget * 4] # Simplistic truncation

        trace = CompressionTrace(
            strategy_name=self.id,
            original_tokens=len(tokenizer(str(text_or_chunks))['input_ids']) if tokenizer else None,
            compressed_tokens=len(tokenizer(compressed_text)['input_ids']) if tokenizer else None,
            # Add more trace details as needed
        )
        return CompressedMemory(text=compressed_text), trace
"""
    (target_dir / "strategy.py").write_text(strategy_py_content)

    manifest = {"package_format_version": "1.0", "strategy_id": name, "strategy_class_name": "MyStrategy", "strategy_module": "strategy", "display_name": name, "version": "0.1.0", "authors": [], "description": "Describe the strategy"}
    (target_dir / "strategy_package.yaml").write_text(yaml.safe_dump(manifest))
    (target_dir / "requirements.txt").write_text("\n"); (target_dir / "README.md").write_text(f"# {name}\n")
    (target_dir / "experiments" / "example.yaml").write_text("""dataset: example.txt\nparam_grid:\n- {}\npackaged_strategy_config:\n  strategy_params: {}\n""")
    typer.echo(f"Successfully created strategy package '{name}' at: {target_dir}")

@dev_app.command("validate-strategy-package", help="Validates the structure and manifest of a compression strategy extension package.\n\nUsage Examples:\n  gist-memory dev validate-strategy-package path/to/my_strategy_pkg")
def validate_strategy_package(package_path: Path = typer.Argument(..., help="Path to the root directory of the strategy package.", exists=True, file_okay=False, dir_okay=True, resolve_path=True)) -> None:
    errors, warnings = validate_package_dir(package_path)
    for w in warnings: typer.secho(f"Warning: {w}", fg=typer.colors.YELLOW)
    if errors:
        for e in errors: typer.echo(e)
        raise typer.Exit(code=1)
    typer.echo(f"Strategy package at '{package_path}' appears valid.")

@dev_app.command("run-package-experiment", help="Runs an experiment defined within a compression strategy extension package.\n\nUsage Examples:\n  gist-memory dev run-package-experiment path/to/my_package --experiment main_test.yaml")
def run_package_experiment(
    package_path: Path = typer.Argument(..., help="Path to the root directory of the strategy package.", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    experiment: Optional[str] = typer.Option(None, "--experiment", help="Name or relative path of the experiment configuration YAML file within the package's 'experiments' directory. If not specified, attempts to run a default experiment if defined in manifest.")
) -> None:
    manifest = load_manifest(package_path / "strategy_package.yaml"); Strategy = load_strategy_class(package_path, manifest)
    missing = check_requirements_installed(package_path / "requirements.txt")
    if missing: typer.secho(f"Warning: The following package requirements are not installed: {', '.join(missing)}. This may cause errors.", fg=typer.colors.YELLOW)

    if experiment is None:
        defaults = manifest.get("default_experiments", [])
        if len(defaults) == 1: experiment = defaults[0].get("path")
    if experiment is None: typer.secho("Error: No experiment specified and no single default experiment found in manifest.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    exp_path = Path(experiment)
    if not exp_path.is_absolute(): exp_path = package_path / "experiments" / exp_path # Assume relative to 'experiments' dir

    if not exp_path.exists(): typer.secho(f"Error: Experiment file '{exp_path}' not found.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    try: cfg_data = yaml.safe_load(exp_path.read_text()) or {}
    except Exception as e: typer.secho(f"Error loading experiment config '{exp_path}': {e}", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    dataset_path_str = cfg_data.get("dataset")
    if not dataset_path_str: typer.secho(f"Error: 'dataset' not specified in experiment config '{exp_path}'.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)

    dataset_path = Path(dataset_path_str)
    if not dataset_path.is_absolute(): dataset_path = (package_path / dataset_path_str).resolve()

    if not dataset_path.exists(): typer.secho(f"Error: Dataset path '{dataset_path}' (resolved from '{dataset_path_str}') not found.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1)
    cfg_data["dataset"] = str(dataset_path)

    params = cfg_data.get("param_grid", [{}])
    cfg = ResponseExperimentConfig(dataset=Path(cfg_data["dataset"]), param_grid=params, validation_metrics=cfg_data.get("validation_metrics"))
    strategy_params = cfg_data.get("packaged_strategy_config", {}).get("strategy_params", {}); strategy = Strategy(**strategy_params)

    typer.echo(f"Running experiment '{exp_path.name}' from package '{package_path.name}' with strategy '{manifest.get('strategy_id', 'Unknown')}'...")
    results = run_response_experiment(cfg, strategy=strategy)
    typer.echo("\n--- Experiment Results ---")
    typer.echo(json.dumps(results, indent=2))

@dev_app.command("run-hpo-script", help="Executes a Python script, typically for Hyperparameter Optimization (HPO).\n\nUsage Examples:\n  gist-memory dev run-hpo-script path/to/my_hpo_optimizer.py")
def run_hpo_script(script_path: Path = typer.Argument(..., help="Path to the Python HPO script to execute.", exists=True, file_okay=True, dir_okay=False, resolve_path=True)) -> None:
    typer.echo(f"Executing HPO script: {script_path}...")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        typer.echo(f"Successfully executed HPO script: {script_path}")
    except Exception as e:
        typer.secho(f"Error executing HPO script '{script_path}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@dev_app.command("inspect-trace", help="Inspects a CompressionTrace JSON file, optionally filtering by step type.")
def inspect_trace(
    trace_file: Path = typer.Argument(..., help="Path to the CompressionTrace JSON file.", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    step_type: Optional[str] = typer.Option(None, "--type", help="Filter trace steps by this 'type' string (e.g., 'chunking', 'llm_call').")
) -> None:
    if not trace_file.exists(): typer.secho(f"Error: Trace file '{trace_file}' not found.", err=True, fg=typer.colors.RED); raise typer.Exit(code=1) # Redundant due to Typer's exists=True, but good practice
    data = json.loads(trace_file.read_text()); steps = data.get("steps", [])

    title = f"Compression Trace: {trace_file.name}"
    if data.get('strategy_name'): title += f" (Strategy: {data['strategy_name']})"
    if data.get('original_tokens'): title += f" | Original Tokens: {data['original_tokens']}"
    if data.get('compressed_tokens'): title += f" | Compressed Tokens: {data['compressed_tokens']}"
    if data.get('processing_ms'): title += f" | Time: {data['processing_ms']:.2f}ms"

    table = Table("Index", "Type", "Details Preview", title=title)
    table = Table("idx", "type", "details")
    for idx, step in enumerate(steps):
        stype = step.get("type")
        if step_type and stype != step_type: continue
        preview = json.dumps(step.get("details", {}))[:50]; table.add_row(str(idx), stype or "", preview)
    console.print(table)

# --- Config App Commands ---
@config_app.command("set", help="Sets a Gist Memory configuration key to a new value in the user's global config file.\n\nUsage Examples:\n  gist-memory config set default_model_id openai/gpt-4-turbo\n  gist-memory config set gist_memory_path /mnt/my_data/gist_memory_store")
def config_set_command(
    ctx: typer.Context,
    key: str = typer.Argument(..., help=f"The configuration key to set. Valid keys: {', '.join(Config.DEFAULT_CONFIG.keys())}."),
    value: str = typer.Argument(..., help="The new value for the configuration key."),
) -> None:
    config: Config = ctx.obj['config']
    try:
        success = config.set(key, value)
        if success:
            typer.secho(f"Successfully set '{key}' to '{value}' in the user global configuration: {Config.USER_CONFIG_PATH}", fg=typer.colors.GREEN)
            typer.echo(f"Note: Environment variables or local project '.gmconfig.yaml' may override this global setting.")
        else:
            # config.set already prints the specific error.
            raise typer.Exit(code=1)
    except Exception as e: # Catch any unexpected errors during the process
        typer.secho(f"An unexpected error occurred while setting configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@config_app.command("show", help="Displays current Gist Memory configuration values, their effective settings, and their sources.\n\nUsage Examples:\n  gist-memory config show\n  gist-memory config show --key default_strategy_id")
def config_show_command(
    ctx: typer.Context,
    key: Optional[str] = typer.Option(None, "--key", "-k", help=f"Specific configuration key to display. Valid keys: {', '.join(Config.DEFAULT_CONFIG.keys())}.")
) -> None:
    config: Config = ctx.obj['config']
    console = Console()
    table = Table(title="Gist Memory Configuration")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Effective Value", style="magenta")
    table.add_column("Source", style="green")

    if key:
        value, source_info = config.get_with_source(key)
        if value is not None:
            table.add_row(key, str(value), source_info)
        else:
            typer.secho(f"Configuration key '{key}' not found or not set.", fg=typer.colors.YELLOW)
            # Optionally, show known keys if a specific one isn't found
            # typer.echo("Known configuration keys are:")
            # for known_key in config.DEFAULT_CONFIG.keys():
            #     typer.echo(f"- {known_key}")
            # raise typer.Exit(code=1) # Or just print the empty table / message
    else:
        all_configs_with_sources = config.get_all_with_sources()
        if not all_configs_with_sources:
            typer.echo("No configurations found.")
            return

        sorted_keys = sorted(all_configs_with_sources.keys())

        for k_val in sorted_keys: # Renamed key to k_val to avoid conflict
            value, source_info = all_configs_with_sources[k_val]
            table.add_row(k_val, str(value) if value is not None else "Not Set", source_info)

    if table.row_count > 0:
        console.print(table)
    elif not key: # only print if not searching for a specific key that wasn't found
        typer.echo("No configuration settings found.")


# Commands still under @app.command that need to be moved or have their functionality subsumed
# @app.command()
# def validate_cmd(...) # This should be agent_app.command("validate")
# @app.command("download-model") -> now dev_app.command("download-embedding-model")
# @app.command("download-chat-model") -> now dev_app.command("download-chat-model")

if __name__ == "__main__":
	app()
