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
from .token_utils import token_count
from rich.table import Table
from rich.console import Console

from compact_memory import __version__
from .logging_utils import configure_logging


from .agent import Agent
from .vector_store import InMemoryVectorStore
from .active_memory_manager import ActiveMemoryManager
from .validation import (
    _VALIDATION_METRIC_REGISTRY,
    get_validation_metric_class,
)
from .model_utils import (
    download_embedding_model as util_download_embedding_model,
    download_chat_model as util_download_chat_model,
)
from .embedding_pipeline import (
    get_embedding_dim,
    EmbeddingDimensionMismatchError,
)
from .utils import load_agent
from compact_memory.config import Config, DEFAULT_CONFIG, USER_CONFIG_PATH

from .compression import (
    available_strategies,
    get_compression_strategy,
    all_strategy_metadata,
    get_strategy_metadata,
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

app = typer.Typer(
    help="Compact Memory: A CLI for intelligent information management using memory agents and advanced compression. Ingest, query, and compress information. Manage agent configurations and developer tools."
)
console = Console()

# --- New Command Groups ---
agent_app = typer.Typer(
    help="Manage memory agents: initialize, inspect statistics, validate, and clear."
)
config_app = typer.Typer(
    help="Manage Compact Memory application configuration settings."
)
dev_app = typer.Typer(
    help="Commands for compression strategy developers and researchers."
)

# --- Add New Command Groups to Main App ---
app.add_typer(agent_app, name="agent")
app.add_typer(config_app, name="config")
app.add_typer(dev_app, name="dev")


def version_callback(value: bool):
    if value:
        typer.echo(f"Compact Memory version: {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Path to write debug logs. If not set, logs are not written to file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose (DEBUG level) logging to console and log file (if specified).",
    ),
    memory_path: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the Compact Memory agent directory. Overrides COMPACT_MEMORY_PATH env var and configuration files.",
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="Default model ID for LLM interactions. Overrides COMPACT_MEMORY_DEFAULT_MODEL_ID env var and configuration files.",
    ),
    strategy_id: Optional[str] = typer.Option(
        None,
        "--strategy-id",
        help="Default compression strategy ID. Overrides COMPACT_MEMORY_DEFAULT_STRATEGY_ID env var and configuration files.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the application version and exit.",
    ),
) -> None:
    # Ensure config is loaded early, before other logic might depend on it.
    # It's important that ctx.obj['config'] is populated.
    if ctx.obj is None:
        ctx.obj = {}
    if "config" not in ctx.obj:  # Could be pre-populated by a test harness or similar
        ctx.obj["config"] = Config()

    config: Config = ctx.obj["config"]
    config.validate()

    resolved_log_file = log_file
    resolved_verbose = verbose

    if resolved_log_file:
        level = logging.DEBUG if resolved_verbose else logging.INFO
        configure_logging(resolved_log_file, level)
    elif resolved_verbose:
        logging.basicConfig(level=logging.DEBUG)

    resolved_memory_path = (
        memory_path if memory_path is not None else config.get("compact_memory_path")
    )
    resolved_model_id = (
        model_id if model_id is not None else config.get("default_model_id")
    )
    resolved_strategy_id = (
        strategy_id if strategy_id is not None else config.get("default_strategy_id")
    )

    if resolved_memory_path:
        resolved_memory_path = str(Path(resolved_memory_path).expanduser())

    # Determine if the command is one that requires memory_path to be set
    command_requires_memory_path = True  # Assume true by default
    if ctx.invoked_subcommand:
        # Commands that DON'T require a memory path immediately
        no_mem_path_commands = ["config", "dev"]
        # Top level commands that might also not need it (e.g. if they are just info commands)
        # For now, assume all other top-level commands (like ingest, query, compress) will need it.
        if ctx.invoked_subcommand in no_mem_path_commands:
            command_requires_memory_path = False
        else:  # Check if it's a subcommand of a group that doesn't need it
            parent_command = ctx.invoked_subcommand.split(".")[
                0
            ]  # Simplistic check, assumes one level of nesting
            if parent_command in no_mem_path_commands:
                command_requires_memory_path = False

    if command_requires_memory_path and not resolved_memory_path:
        is_interactive = sys.stdin.isatty()
        prompt_default_path = config.get("compact_memory_path")

        if is_interactive:
            typer.secho("The Compact Memory path is not set.", fg=typer.colors.YELLOW)
            new_path_input = typer.prompt(
                "Please enter the path for Compact Memory storage",
                default=(
                    str(Path(prompt_default_path).expanduser())
                    if prompt_default_path
                    else None
                ),
            )
            if new_path_input:
                resolved_memory_path = str(Path(new_path_input).expanduser())
                typer.secho(
                    f"Using memory path: {resolved_memory_path}", fg=typer.colors.GREEN
                )
                typer.echo(
                    f'To set this path permanently, run: compact-memory config set compact_memory_path "{resolved_memory_path}"'
                )
            else:
                typer.secho(
                    "Memory path is required to proceed.", fg=typer.colors.RED, err=True
                )
                raise typer.Exit(code=1)
        else:
            typer.secho(
                "Error: Compact Memory path is not set.", fg=typer.colors.RED, err=True
            )
            typer.secho(
                "Please set it using the --memory-path option, the COMPACT_MEMORY_PATH environment variable, or in a config file (~/.config/compact_memory/config.yaml or .gmconfig.yaml).",
                err=True,
            )
            raise typer.Exit(code=1)

    load_plugins()

    # ctx.obj['config'] is already set above
    ctx.obj.update(
        {
            "verbose": resolved_verbose,
            "log_file": resolved_log_file,
            "compact_memory_path": resolved_memory_path,
            "default_model_id": resolved_model_id,
            "default_strategy_id": resolved_strategy_id,
            # "config": config, # Already present
        }
    )


class PersistenceLock:
    """Deprecated lock context (no-op)."""

    def __init__(self, path: Path) -> None:  # pragma: no cover - legacy
        self.path = path

    def __enter__(self):  # pragma: no cover - legacy
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - legacy
        return None


def _corrupt_exit(path: Path, exc: Exception) -> None:
    typer.echo(f"Error: Brain data is corrupted. {exc}", err=True)
    typer.echo(
        f"Try running compact-memory validate {path} for more details or restore from a backup.",
        err=True,
    )
    raise typer.Exit(code=1)


def _load_agent(path: Path) -> Agent:
    """Return a new in-memory agent (persistence removed)."""
    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    return Agent(store)


# --- Agent Commands ---
@agent_app.command(
    "init",
    help="Initializes a new Compact Memory agent in a specified directory.\n\nUsage Examples:\n  compact-memory agent init ./my_agent_dir\n  compact-memory agent init /path/to/another_agent --name 'research_agent' --model-name 'sentence-transformers/all-mpnet-base-v2' --tau 0.75",
)
def init(
    target_directory: Path = typer.Argument(
        ...,
        help="Directory to initialize the new agent in. Will be created if it doesn't exist.",
        resolve_path=True,
    ),
    *,
    ctx: typer.Context,
    name: str = typer.Option("default", help="A descriptive name for the agent."),
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2",
        help="Name of the sentence-transformer model for embeddings.",
    ),
    tau: float = typer.Option(
        0.8,
        help="Similarity threshold (tau) for memory consolidation, between 0.5 and 0.95.",
    ),  # Added range to help
    alpha: float = typer.Option(
        0.1, help="Alpha parameter, controlling the decay rate for memory importance."
    ),
    chunker: str = typer.Option(
        "sentence_window",
        help="Chunking strategy to use for processing text during ingestion.",
    ),
) -> None:
    path = target_directory.expanduser()
    if path.exists() and any(path.iterdir()):
        typer.secho(
            f"Error: Directory '{path}' already exists and is not empty.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if not 0.5 <= tau <= 0.95:
        typer.secho(
            "Error: --tau must be between 0.5 and 0.95.", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)  # Keep runtime check
    try:
        dim = get_embedding_dim()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    store = InMemoryVectorStore(embedding_dim=dim)
    store.meta.update(
        {"agent_name": name, "tau": tau, "alpha": alpha, "chunker": chunker}
    )
    store.save()
    typer.echo(f"Successfully initialized Compact Memory agent at {path}")


@agent_app.command(
    "stats",
    help="Displays statistics about the Compact Memory agent.\n\nUsage Examples:\n  compact-memory agent stats\n  compact-memory agent stats --memory-path path/to/my_agent --json",
)
def stats(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the agent directory. Overrides global setting if provided.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output statistics in JSON format."
    ),
) -> None:
    final_memory_path_str = (
        memory_path_arg
        if memory_path_arg is not None
        else ctx.obj.get("compact_memory_path")
    )
    if final_memory_path_str is None:
        typer.secho(
            "Critical Error: Memory path could not be resolved for stats.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists():
        typer.secho(
            f"Error: Memory path '{path}' not found or is invalid.",
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


@agent_app.command("validate", help="Validates the integrity of the agent's storage.")
def validate_agent_storage(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the agent directory. Overrides global setting if provided.",
    ),
) -> None:
    final_memory_path_str = (
        memory_path_arg
        if memory_path_arg is not None
        else ctx.obj.get("compact_memory_path")
    )
    if final_memory_path_str is None:
        typer.secho(
            "Critical Error: Memory path could not be resolved for validate.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists():
        typer.secho(
            f"Error: Memory path '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo("No built-in storage to validate.")


@agent_app.command(
    "clear",
    help="Deletes all data from an agent's memory. This action is irreversible.\n\nUsage Examples:\n  compact-memory agent clear --force\n  compact-memory agent clear --memory-path path/to/another_agent --dry-run",
)
def clear(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the agent directory. Overrides global setting if provided.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force deletion without prompting for confirmation.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate deletion and show what would be deleted without actually removing files.",
    ),
) -> None:
    final_memory_path_str = (
        memory_path_arg
        if memory_path_arg is not None
        else ctx.obj.get("compact_memory_path")
    )
    if final_memory_path_str is None:
        typer.secho(
            "Critical Error: Memory path could not be resolved for clear.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    path = Path(final_memory_path_str)
    if not path.exists():
        typer.secho(
            f"Error: Memory path '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if dry_run:
        typer.echo("Dry run: nothing to delete (no persistence layer).")
        return
    if not force:
        if not typer.confirm(
            f"Are you sure you want to delete all data in agent at '{path}'? This cannot be undone.",
            abort=True,
        ):
            return  # Added path and warning
    if path.exists():
        shutil.rmtree(path)
        typer.echo(f"Successfully cleared agent data at {path}")
    else:
        typer.secho(
            f"Error: Directory '{path}' not found.", err=True, fg=typer.colors.RED
        )  # Added path


# --- Top-Level Commands ---
@app.command(
    "ingest",
    help="Ingests text from a file or directory into the agent's memory.\n\nUsage Examples:\n  compact-memory ingest path/to/my_data.txt --tau 0.7\n  compact-memory ingest path/to/my_directory/",
)
def ingest(
    ctx: typer.Context,
    source: Path = typer.Argument(
        ...,
        help="Path to the text file or directory containing text files to ingest.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
    ),
    tau: Optional[float] = typer.Option(
        None,
        "--tau",
        "-t",
        help="Similarity threshold (0.5-0.95) for memory consolidation.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output ingestion summary statistics in JSON format."
    ),
) -> None:
    """Ingest ``source`` into a temporary in-memory agent and report metrics."""

    texts: list[str] = []
    if source.is_dir():
        for path in source.rglob("*.txt"):
            try:
                texts.append(path.read_text())
            except Exception:
                continue
    else:
        texts.append(source.read_text())

    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    agent = Agent(store, similarity_threshold=tau or 0.8)

    start = time.perf_counter()
    for text in texts:
        agent.add_memory(text)
    duration = time.perf_counter() - start

    metrics = dict(agent.metrics)
    metrics.update(
        {
            "prototype_count": len(agent.store.prototypes),
            "memory_count": len(agent.store.memories),
            "ingest_seconds": duration,
        }
    )

    if json_output:
        typer.echo(json.dumps(metrics))
    else:
        for k, v_val in metrics.items():
            typer.echo(f"{k}: {v_val}")


@app.command(
    "query",
    help='Queries the Compact Memory agent and returns an AI-generated response.\n\nUsage Examples:\n  compact-memory query "What is the capital of France?"\n  compact-memory query "Explain the theory of relativity in simple terms" --show-prompt-tokens',
)
def query(
    ctx: typer.Context,
    query_text: str = typer.Argument(..., help="The query text to send to the agent."),
    show_prompt_tokens: bool = typer.Option(
        False,
        "--show-prompt-tokens",
        help="Display the token count of the final prompt sent to the LLM.",
    ),
) -> None:
    path = Path("./memory")  # unused placeholder for legacy arg
    agent = _load_agent(path)
    final_model_id = ctx.obj.get("default_model_id")  # Renamed for clarity
    final_strategy_id = ctx.obj.get("default_strategy_id")  # Renamed for clarity

    if final_model_id is None:
        typer.secho(
            "Error: Default Model ID not specified. Use --model-id option or set in config.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # LLM integration removed - skip loading chat model

    mgr = ActiveMemoryManager()
    comp_strategy_instance = None  # Renamed for clarity
    if final_strategy_id and final_strategy_id.lower() != "none":
        try:
            comp_cls = get_compression_strategy(final_strategy_id)
            info = get_strategy_metadata(final_strategy_id)
            if info and info.get("source") == "contrib":
                typer.secho(
                    "\u26a0\ufe0f Using experimental strategy from contrib: not officially supported.",
                    fg=typer.colors.YELLOW,
                )
            comp_strategy_instance = comp_cls()
        except KeyError:
            typer.secho(
                f"Error: Unknown compression strategy '{final_strategy_id}' (from global config/option). Available: {', '.join(available_strategies())}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)  # Added available strategies

    try:
        result = agent.receive_channel_message(
            "cli", query_text, mgr, compression=comp_strategy_instance
        )
    except TypeError:  # Older agents might not have compression param
        result = agent.receive_channel_message("cli", query_text, mgr)

    reply = result.get("reply")
    if reply:
        typer.echo(reply)
    else:
        typer.secho("The agent did not return a reply.", fg=typer.colors.YELLOW)

    if show_prompt_tokens and result.get("prompt_tokens") is not None:
        typer.echo(f"Prompt tokens: {result['prompt_tokens']}")


@app.command(
    "compress",
    help='Compress text using a specified strategy. Compresses text from a string, file, or directory.\n\nUsage Examples:\n  compact-memory compress --text "Some very long text..." --strategy first_last --budget 100\n  compact-memory compress --file path/to/document.txt -s prototype -b 200 -o summary.txt\n  compact-memory compress --dir input_dir/ -s custom_package_strat -b 500 --output-dir output_dir/ --recursive -p "*.md"',
)
def compress(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(
        None,
        "--text",
        help="Raw text to compress or '-' to read from stdin",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        exists=True,
        dir_okay=False,
        help="Path to a single text file",
    ),
    dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        exists=True,
        file_okay=False,
        help="Path to a directory of input files",
    ),
    *,
    strategy_arg: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Compression strategy ID to use. Overrides the global default strategy.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        resolve_path=True,
        help="File path to write compressed output. If unspecified, prints to console.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        resolve_path=True,
        help="Directory to write compressed files when --dir is used.",
    ),
    output_trace: Optional[Path] = typer.Option(
        None,
        "--output-trace",
        resolve_path=True,
        help="File path to write the CompressionTrace JSON object. (Not applicable for directory input).",
    ),
    recursive: bool = typer.Option(
        False,
        "-r",
        "--recursive",
        help="Process text files in subdirectories recursively when --dir is used.",
    ),
    pattern: str = typer.Option(
        "*.txt",
        "-p",
        "--pattern",
        help="File glob pattern to match files when --dir is used (e.g., '*.md', '**/*.txt').",
    ),
    budget: int = typer.Option(
        ...,
        help="Token budget for the compressed output. The strategy will aim to keep the output within this limit.",
    ),
    verbose_stats: bool = typer.Option(
        False,
        "--verbose-stats",
        help="Show detailed token counts and processing time per item.",
    ),
) -> None:
    final_strategy_id = (
        strategy_arg if strategy_arg is not None else ctx.obj.get("default_strategy_id")
    )
    if not final_strategy_id:
        typer.secho(
            "Error: Compression strategy not specified. Use --strategy option or set COMPACT_MEMORY_DEFAULT_STRATEGY_ID / config.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokenizer = lambda text, **k: {"input_ids": enc.encode(text)}
    except Exception:
        tokenizer = lambda t, **k: t.split()

    if sum(x is not None for x in (text, file, dir)) != 1:
        typer.echo("Error: specify exactly ONE of --text / --file / --dir", err=True)
        raise typer.Exit(1)

    if dir is None:
        if output_dir is not None:
            typer.echo("--output-dir is only valid with --dir", err=True)
            raise typer.Exit(1)
        if recursive:
            typer.echo("--recursive is only valid with --dir", err=True)
            raise typer.Exit(1)
        if pattern != "*.txt":
            typer.echo("--pattern is only valid with --dir", err=True)
            raise typer.Exit(1)
    else:
        if output_trace is not None:
            typer.echo("--output-trace is not valid with --dir", err=True)
            raise typer.Exit(1)

    if text is not None:
        if text == "-":
            text = sys.stdin.read()
        _process_string_compression(
            text,
            final_strategy_id,
            budget,
            output_file,
            output_trace,
            verbose_stats,
            tokenizer,
        )
    elif file is not None:
        _process_file_compression(
            file,
            final_strategy_id,
            budget,
            output_file,
            verbose_stats,
            tokenizer,
            output_trace,
        )
    else:
        if "**" in pattern and not recursive:
            typer.secho(
                "Pattern includes '**' but --recursive not set; matching may miss subdirectories",
                fg=typer.colors.YELLOW,
            )
        _process_directory_compression(
            dir,
            final_strategy_id,
            budget,
            output_dir,
            recursive,
            pattern,
            verbose_stats,
            tokenizer,
            None,
        )


# --- Helper functions for compress (formerly summarize/compress_text) ---
def _process_string_compression(
    text_content: str,
    strategy_id: str,
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
) -> None:
    try:
        strat_cls = get_compression_strategy(strategy_id)
        info = get_strategy_metadata(strategy_id)
        if info and info.get("source") == "contrib":
            typer.secho(
                "\u26a0\ufe0f Using experimental strategy from contrib: not officially supported.",
                fg=typer.colors.YELLOW,
            )
    except KeyError:
        typer.secho(
            f"Unknown compression strategy '{strategy_id}'. Available: {', '.join(available_strategies())}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    strat = strat_cls()
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
            typer.secho(
                f"Error writing {output_file}: {exc}", err=True, fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        typer.echo(f"Saved compressed output to {output_file}")
    else:
        typer.echo(compressed.text)
    if trace_file and trace:
        try:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(json.dumps(asdict(trace)))
        except (IOError, OSError, PermissionError) as exc:
            typer.secho(
                f"Error writing {trace_file}: {exc}", err=True, fg=typer.colors.RED
            )
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
    if not file_path.exists() or not file_path.is_file():
        typer.secho(f"File not found: {file_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        text_content = file_path.read_text()
    except (IOError, OSError, PermissionError) as exc:
        typer.secho(f"Error reading {file_path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _process_string_compression(
        text_content,
        strategy_id,
        budget,
        output_file,
        trace_file,
        verbose_stats,
        tokenizer,
    )


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
    if not dir_path.exists() or not dir_path.is_dir():
        typer.secho(f"Directory not found: {dir_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    files = list(dir_path.rglob(pattern) if recursive else dir_path.glob(pattern))
    if not files:
        typer.echo("No matching files found.")
        return
    if trace_file is not None:
        typer.secho(
            "--output-trace ignored for directory input", fg=typer.colors.YELLOW
        )
        trace_file = None
    count = 0
    for input_file in files:
        typer.echo(f"Processing {input_file}...")
        if output_dir_param:
            try:
                output_dir_param.mkdir(parents=True, exist_ok=True)
            except (IOError, OSError, PermissionError) as exc:
                typer.secho(
                    f"Error creating {output_dir_param}: {exc}",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            rel = input_file.relative_to(dir_path)
            out_path = output_dir_param / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = input_file.with_name(
                f"{input_file.stem}_compressed{input_file.suffix}"
            )
        _process_file_compression(
            input_file, strategy_id, budget, out_path, verbose_stats, tokenizer, None
        )
        count += 1
    if verbose_stats:
        typer.echo(f"Processed {count} files.")


# --- Dev App Commands ---
@dev_app.command(
    "list-metrics",
    help="Lists all available validation metric IDs that can be used in evaluations.",
)
def list_metrics() -> None:
    """Lists all registered validation metric IDs."""
    if not _VALIDATION_METRIC_REGISTRY:
        typer.echo("No validation metrics found.")
        return
    typer.echo("Available validation metric IDs:")
    for mid in sorted(_VALIDATION_METRIC_REGISTRY):
        typer.echo(f"- {mid}")


@dev_app.command(
    "list-strategies",
    help="Lists all available compression strategy IDs, their versions, and sources (built-in or plugin).",
)
def list_strategies(
    include_contrib: bool = typer.Option(
        False, "--include-contrib", help="Include experimental contrib strategies."
    )
) -> None:
    """Displays a table of registered compression strategies."""
    load_plugins()  # Ensure plugins are loaded
    if include_contrib:
        try:
            from contrib import enable_all_contrib_strategies

            enable_all_contrib_strategies()
        except Exception:  # pragma: no cover - contrib may be missing
            pass
    table = Table(
        "Strategy ID",
        "Display Name",
        "Version",
        "Source",
        "Status",
        title="Available Compression Strategies",
    )
    meta = all_strategy_metadata()
    ids = available_strategies()
    if not ids:
        typer.echo("No compression strategies found.")
        return
    for sid in sorted(ids):
        info = meta.get(sid, {})
        if not include_contrib and info.get("source") == "contrib":
            continue
        status = ""
        if info.get("overrides"):
            status = f"Overrides '{info['overrides']}'"
        table.add_row(
            sid,
            info.get("display_name", sid) or sid,
            info.get("version", "N/A") or "N/A",
            info.get("source", "built-in") or "built-in",
            status,
        )
    console.print(table)


@dev_app.command(
    "inspect-strategy",
    help="Inspects aspects of a compression strategy, currently focused on 'prototype' strategy's beliefs.",
)
def inspect_strategy(
    strategy_name: str = typer.Argument(
        ...,
        help="The name of the strategy to inspect. Currently, only 'prototype' is supported.",
    ),
    *,
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the agent directory. Overrides global setting if provided. Required if 'list-prototypes' is used.",
    ),
    list_prototypes: bool = typer.Option(
        False,
        "--list-prototypes",
        help="List consolidated prototypes (beliefs) if the strategy is 'prototype' and an agent path is provided.",
    ),
) -> None:
    if strategy_name.lower() != "prototype":
        typer.secho(
            f"Error: Inspection for strategy '{strategy_name}' is not supported. Only 'prototype' is currently inspectable.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if list_prototypes:
        final_memory_path_str = (
            memory_path_arg
            if memory_path_arg is not None
            else ctx.obj.get("compact_memory_path")
        )
        if not final_memory_path_str:
            typer.secho(
                "Error: --memory-path is required when --list-prototypes is used.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        path = Path(final_memory_path_str)
        if not path.exists():
            typer.secho(
                f"Error: Compact Memory path '{path}' not found or is invalid.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        agent = _load_agent(path)
        protos = agent.get_prototypes_view()
        if not protos:
            typer.echo(f"No prototypes found in agent at '{path}'.")
            return
        table = Table(
            "ID",
            "Strength",
            "Confidence",
            "Summary",
            title=f"Prototypes for Agent at '{path}'",
        )
        for p in protos:
            table.add_row(
                p["id"],
                f"{p['strength']:.2f}",
                f"{p['confidence']:.2f}",
                p["summary"][:80],
            )  # Increased summary length
        console.print(table)
    else:
        typer.echo(
            f"Strategy '{strategy_name}' is available. Use --list-prototypes and provide an agent path to see its beliefs."
        )


@dev_app.command(
    "evaluate-compression",
    help='Evaluates compressed text against original text using a specified metric.\n\nUsage Examples:\n  compact-memory dev evaluate-compression original.txt summary.txt --metric compression_ratio\n  echo "original text" | compact-memory dev evaluate-compression - summary.txt --metric some_other_metric --metric-params \'{"param": "value"}\'',
)
def evaluate_compression_cmd(
    original_input: str = typer.Argument(
        ...,
        help="Original text content, path to a text file, or '-' to read from stdin.",
    ),
    compressed_input: str = typer.Argument(
        ...,
        help="Compressed text content, path to a text file, or '-' to read from stdin.",
    ),
    metric_id: str = typer.Option(
        ...,
        "--metric",
        "-m",
        help="ID of the validation metric to use (see 'list-metrics').",
    ),
    metric_params_json: Optional[str] = typer.Option(
        None,
        "--metric-params",
        help='Metric parameters as a JSON string (e.g., \'{"model_name": "bert-base-uncased"}\').',
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output evaluation scores in JSON format."
    ),
) -> None:
    def read_input(value: str, allow_stdin: bool) -> str:
        if value == "-":
            if not allow_stdin:
                typer.secho(
                    "Error: Cannot use '-' for stdin for both original and compressed input simultaneously.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(value)
        if p.exists() and p.is_file():
            try:
                return p.read_text()
            except Exception as e:
                typer.secho(
                    f"Error reading file '{p}': {e}", err=True, fg=typer.colors.RED
                )
                raise typer.Exit(code=1)
        return value  # Treat as direct text content if not a file or '-'

    orig_text = read_input(original_input, True)
    comp_text = read_input(
        compressed_input, original_input != "-"
    )  # Only allow stdin for compressed if original isn't stdin

    try:
        Metric = get_validation_metric_class(metric_id)
    except KeyError:
        typer.secho(
            f"Error: Unknown metric ID '{metric_id}'. Use 'list-metrics' to see available IDs.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    params = {}
    if metric_params_json:
        try:
            params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc:
            typer.secho(
                f"Error: Invalid JSON in --metric-params: {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    metric = Metric(**params)
    try:
        scores = metric.evaluate(original_text=orig_text, compressed_text=comp_text)
    except Exception as exc:
        typer.secho(
            f"Error during metric evaluation: {exc}", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(scores))
    else:
        typer.echo(f"Scores for metric '{metric_id}':")
        for k, v in scores.items():
            typer.echo(f"- {k}: {v}")


@dev_app.command(
    "evaluate-llm-response",
    help="Evaluates an LLM's response against a reference answer using a specified metric.",
)
def evaluate_llm_response_cmd(
    response_input: str = typer.Argument(
        ...,
        help="LLM's generated response text, path to a response file, or '-' to read from stdin.",
    ),
    reference_input: str = typer.Argument(
        ..., help="Reference (ground truth) answer text or path to a reference file."
    ),
    metric_id: str = typer.Option(
        ...,
        "--metric",
        "-m",
        help="ID of the validation metric to use (see 'list-metrics').",
    ),
    metric_params_json: Optional[str] = typer.Option(
        None,
        "--metric-params",
        help='Metric parameters as a JSON string (e.g., \'{"model_name": "bert-base-uncased"}\').',
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output evaluation scores in JSON format."
    ),
) -> None:
    def read_value(
        val: str, input_name: str, allow_stdin: bool
    ) -> str:  # Added input_name
        if val == "-":
            if not allow_stdin:
                typer.secho(
                    f"Error: Cannot use '-' for stdin for both response and reference input simultaneously.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            return sys.stdin.read()
        p = Path(val)
        if p.exists() and p.is_file():
            try:
                return p.read_text()
            except Exception as e:
                typer.secho(
                    f"Error reading {input_name} file '{p}': {e}",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
        return val  # Treat as direct content

    resp_text = read_value(response_input, "LLM response", True)
    ref_text = read_value(reference_input, "reference answer", response_input != "-")

    try:
        Metric = get_validation_metric_class(metric_id)
    except KeyError:
        typer.secho(
            f"Error: Unknown metric ID '{metric_id}'. Use 'list-metrics' to see available IDs.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    params = {}
    if metric_params_json:
        try:
            params = json.loads(metric_params_json)
        except json.JSONDecodeError as exc:
            typer.secho(
                f"Error: Invalid JSON in --metric-params: {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    metric = Metric(**params)
    try:
        scores = metric.evaluate(llm_response=resp_text, reference_answer=ref_text)
    except Exception as exc:
        typer.secho(
            f"Error during metric evaluation: {exc}", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(scores))
    else:
        typer.echo(f"Scores for metric '{metric_id}':")
        for k, v in scores.items():
            typer.echo(f"- {k}: {v}")


@dev_app.command(
    "download-embedding-model",
    help="Downloads a specified SentenceTransformer embedding model from Hugging Face.",
)
def download_embedding_model_cli(
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2",
        help="Name of the SentenceTransformer model to download (e.g., 'all-MiniLM-L6-v2').",
    )
) -> None:
    typer.echo(f"Starting download for embedding model: {model_name}...")
    bar = tqdm(total=1, desc=f"Downloading {model_name}", unit="model", disable=False)
    try:
        util_download_embedding_model(model_name)
        bar.update(1)
        typer.echo(f"Successfully downloaded embedding model '{model_name}'.")
    except Exception as e:
        typer.secho(
            f"Error downloading embedding model '{model_name}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    finally:
        bar.close()


@dev_app.command(
    "download-chat-model",
    help="Downloads a specified causal Language Model (e.g., for chat) from Hugging Face.",
)
def download_chat_model_cli(
    model_name: str = typer.Option(
        "tiny-gpt2",
        help="Name of the Hugging Face causal LM to download (e.g., 'gpt2', 'facebook/opt-125m').",
    )
) -> None:
    typer.echo(f"Starting download for chat model: {model_name}...")
    bar = tqdm(total=1, desc=f"Downloading {model_name}", unit="model", disable=False)
    try:
        util_download_chat_model(model_name)
        bar.update(1)
        typer.echo(f"Successfully downloaded chat model '{model_name}'.")
    except Exception as e:
        typer.secho(
            f"Error downloading chat model '{model_name}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    finally:
        bar.close()


@dev_app.command(
    "create-strategy-package",
    help="Creates a new compression strategy extension package from a template. This command generates a template directory with all the necessary files to start developing a new, shareable strategy package, including a sample strategy, manifest file, and README.",
)
def create_strategy_package(
    name: str = typer.Option(
        "sample_strategy",
        "--name",
        help="Name for the new strategy (e.g., 'my_custom_strategy'). Used for directory and strategy ID.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        help="Directory where the strategy package will be created. Defaults to a new directory named after the strategy in the current location.",
    ),
) -> None:
    target_dir = Path(path or name).resolve()  # Renamed for clarity

    if target_dir.exists() and any(target_dir.iterdir()):
        typer.secho(
            f"Error: Output directory '{target_dir}' already exists and is not empty.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "experiments").mkdir(exist_ok=True)

    strategy_py_content = f"""from compact_memory.compression.strategies_abc import CompressionStrategy, CompressedMemory, CompressionTrace
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
    (target_dir / "strategy_package.yaml").write_text(yaml.safe_dump(manifest))
    (target_dir / "requirements.txt").write_text("\n")
    (target_dir / "README.md").write_text(f"# {name}\n")
    (target_dir / "experiments" / "example.yaml").write_text(
        """dataset: example.txt\nparam_grid:\n- {}\npackaged_strategy_config:\n  strategy_params: {}\n"""
    )
    typer.echo(f"Successfully created strategy package '{name}' at: {target_dir}")


@dev_app.command(
    "validate-strategy-package",
    help="Validates the structure and manifest of a compression strategy extension package.\n\nUsage Examples:\n  compact-memory dev validate-strategy-package path/to/my_strategy_pkg",
)
def validate_strategy_package(
    package_path: Path = typer.Argument(
        ...,
        help="Path to the root directory of the strategy package.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    )
) -> None:
    errors, warnings = validate_package_dir(package_path)
    for w in warnings:
        typer.secho(f"Warning: {w}", fg=typer.colors.YELLOW)
    if errors:
        for e in errors:
            typer.echo(e)
        raise typer.Exit(code=1)
    typer.echo(f"Strategy package at '{package_path}' appears valid.")


@dev_app.command(
    "run-hpo-script",
    help="Executes a Python script, typically for Hyperparameter Optimization (HPO).\n\nUsage Examples:\n  compact-memory dev run-hpo-script path/to/my_hpo_optimizer.py",
)
def run_hpo_script(
    script_path: Path = typer.Argument(
        ...,
        help="Path to the Python HPO script to execute.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    )
) -> None:
    typer.echo(f"Executing HPO script: {script_path}...")
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        typer.echo(f"Successfully executed HPO script: {script_path}")
    except Exception as e:
        typer.secho(
            f"Error executing HPO script '{script_path}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@dev_app.command(
    "inspect-trace",
    help="Inspects a CompressionTrace JSON file, optionally filtering by step type.",
)
def inspect_trace(
    trace_file: Path = typer.Argument(
        ...,
        help="Path to the CompressionTrace JSON file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    step_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Filter trace steps by this 'type' string (e.g., 'chunking', 'llm_call').",
    ),
) -> None:
    if not trace_file.exists():
        typer.secho(
            f"Error: Trace file '{trace_file}' not found.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(
            code=1
        )  # Redundant due to Typer's exists=True, but good practice
    data = json.loads(trace_file.read_text())
    steps = data.get("steps", [])

    title = f"Compression Trace: {trace_file.name}"
    if data.get("strategy_name"):
        title += f" (Strategy: {data['strategy_name']})"
    if data.get("original_tokens"):
        title += f" | Original Tokens: {data['original_tokens']}"
    if data.get("compressed_tokens"):
        title += f" | Compressed Tokens: {data['compressed_tokens']}"
    if data.get("processing_ms"):
        title += f" | Time: {data['processing_ms']:.2f}ms"

    console.print(f"Strategy: {data.get('strategy_name', '')}")
    table = Table("Index", "Type", "Details Preview", title=title)
    for idx, step in enumerate(steps):
        stype = step.get("type")
        if step_type and stype != step_type:
            continue
        preview = json.dumps(step.get("details", {}))[:50]
        table.add_row(str(idx), stype or "", preview)
    console.print(table)


# --- Config App Commands ---
@config_app.command(
    "set",
    help="Sets a Compact Memory configuration key to a new value in the user's global config file.\n\nUsage Examples:\n  compact-memory config set default_model_id openai/gpt-4-turbo\n  compact-memory config set compact_memory_path /mnt/my_data/compact_memory_store",
)
def config_set_command(
    ctx: typer.Context,
    key: str = typer.Argument(
        ...,
        help=f"The configuration key to set. Valid keys: {', '.join(DEFAULT_CONFIG.keys())}.",
    ),
    value: str = typer.Argument(..., help="The new value for the configuration key."),
) -> None:
    config: Config = ctx.obj["config"]
    try:
        success = config.set(key, value)
        if success:
            typer.secho(
                f"Successfully set '{key}' to '{value}' in the user global configuration: {USER_CONFIG_PATH}",
                fg=typer.colors.GREEN,
            )
            typer.echo(
                f"Note: Environment variables or local project '.gmconfig.yaml' may override this global setting."
            )
        else:
            # config.set already prints the specific error.
            raise typer.Exit(code=1)
    except Exception as e:  # Catch any unexpected errors during the process
        typer.secho(
            f"An unexpected error occurred while setting configuration: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@config_app.command(
    "show",
    help="Displays current Compact Memory configuration values, their effective settings, and their sources.\n\nUsage Examples:\n  compact-memory config show\n  compact-memory config show --key default_strategy_id",
)
def config_show_command(
    ctx: typer.Context,
    key: Optional[str] = typer.Option(
        None,
        "--key",
        "-k",
        help=f"Specific configuration key to display. Valid keys: {', '.join(DEFAULT_CONFIG.keys())}.",
    ),
) -> None:
    config: Config = ctx.obj["config"]
    console = Console(width=200)
    table = Table(title="Compact Memory Configuration")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Effective Value", style="magenta", overflow="fold")
    table.add_column("Source", style="green", no_wrap=True, overflow="fold")

    if key:
        value, source_info = config.get_with_source(key)
        if value is not None:
            table.add_row(key, str(value), source_info)
        else:
            typer.secho(
                f"Configuration key '{key}' not found or not set.",
                fg=typer.colors.YELLOW,
            )
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

        for k_val in sorted_keys:  # Renamed key to k_val to avoid conflict
            value, source_info = all_configs_with_sources[k_val]
            table.add_row(k_val, str(value), source_info)

    if table.row_count > 0:
        console.print(table)
    elif not key:  # only print if not searching for a specific key that wasn't found
        typer.echo("No configuration settings found.")


# Commands still under @app.command that need to be moved or have their functionality subsumed
# @app.command()
# def validate_cmd(...) # This should be agent_app.command("validate")
# @app.command("download-model") -> now dev_app.command("download-embedding-model")
# @app.command("download-chat-model") -> now dev_app.command("download-chat-model")

if __name__ == "__main__":
    app()
