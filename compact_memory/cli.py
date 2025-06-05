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

import typer
from .token_utils import token_count
from rich.table import Table
from rich.console import Console

from compact_memory import __version__
from .logging_utils import configure_logging


from .memory_container import MemoryContainer
from .vector_store import InMemoryVectorStore
from . import local_llm
from CompressionEngine.contrib import ( # Updated path
    ActiveMemoryManager, # Assuming this class remains or its new equivalent is imported
    enable_all_experimental_engines, # Updated function name
)
from .validation.registry import (
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
from compact_memory.config import Config, DEFAULT_CONFIG, USER_CONFIG_PATH # Config class might need internal update for default_engine_id

from CompressionEngine.core.registry import ( # Updated path
    available_engines, # Updated function name
    get_compression_engine, # Updated function name
    all_engine_metadata, # Updated function name
    get_engine_metadata, # Updated function name
)
from .plugin_loader import load_plugins # plugin_loader now uses 'engines' internally
from .cli_plugins import load_cli_plugins
from .package_utils import ( # package_utils functions need internal updates too
    load_manifest,
    validate_manifest,
    load_engine_class_from_module as load_engine_class, # Updated function name and alias
    validate_package_dir,
    check_requirements_installed,
)


app = typer.Typer(
    help="Compact Memory: manage stored context containers and advanced compression. Ingest, query, and compress information. Manage container configurations and developer tools."
)
console = Console()

# --- New Command Groups ---
memory_app = typer.Typer(
    help="Manage memory containers: initialize, inspect statistics, validate, and clear."
)
config_app = typer.Typer(
    help="Manage Compact Memory application configuration settings."
)
dev_app = typer.Typer(
    help="Commands for compression engine developers and researchers."
)

# --- Add New Command Groups to Main App ---
app.add_typer(memory_app, name="memory")
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
        help="Path to the Compact Memory memory container directory. Overrides COMPACT_MEMORY_PATH env var and configuration files.",
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="Default model ID for LLM interactions. Overrides COMPACT_MEMORY_DEFAULT_MODEL_ID env var and configuration files.",
    ),
    engine_id: Optional[str] = typer.Option( # Updated option name
        None,
        "--engine-id", # Updated option name
        help="Default compression engine ID. Overrides COMPACT_MEMORY_DEFAULT_ENGINE_ID env var and configuration files.", # Updated help
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
    resolved_engine_id = ( # Updated variable name
        engine_id if engine_id is not None else config.get("default_engine_id") # Updated option name and config key
    )

    if resolved_memory_path:
        resolved_memory_path = str(Path(resolved_memory_path).expanduser())

    # Determine if the command is one that requires memory_path to be set
    command_requires_memory_path = True  # Assume true by default
    if ctx.invoked_subcommand:
        # Commands that DON'T require a memory path immediately
        no_mem_path_commands = ["config", "dev", "compress"]
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

    load_plugins() # plugin_loader has been updated to find engines
    enable_all_experimental_engines() # Updated function name

    # ctx.obj['config'] is already set above
    ctx.obj.update(
        {
            "verbose": resolved_verbose,
            "log_file": resolved_log_file,
            "compact_memory_path": resolved_memory_path,
            "default_model_id": resolved_model_id,
            "default_engine_id": resolved_engine_id, # Updated key and variable
            # "config": config, # Already present
        }
    )


def _corrupt_exit(path: Path, exc: Exception) -> None:
    typer.echo(f"Error: Brain data is corrupted. {exc}", err=True)
    typer.echo(
        f"Try running compact-memory validate {path} for more details or restore from a backup.",
        err=True,
    )
    raise typer.Exit(code=1)


def load_memory_container(path: Path) -> MemoryContainer:
    """Placeholder loader since on-disk storage was removed."""
    raise RuntimeError(
        "Persistent storage support was removed. Provide your own loader."
    )


# --- Memory Commands ---
@memory_app.command(
    "init",
    help="Initializes a new Compact Memory memory container in a specified directory.\n\nUsage Examples:\n  compact-memory memory init ./my_memory_dir\n  compact-memory memory init /path/to/another_container --name 'research_mem' --model-name 'sentence-transformers/all-mpnet-base-v2' --tau 0.75",
)
def init(
    target_directory: Path = typer.Argument(
        ...,
        help="Directory to initialize the new memory container in. Will be created if it doesn't exist.",
        resolve_path=True,
    ),
    *,
    ctx: typer.Context,
    name: str = typer.Option(
        "default", help="A descriptive name for the memory container."
    ),
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
        {"memory_name": name, "tau": tau, "alpha": alpha, "chunker": chunker}
    )
    store.save()
    typer.echo(f"Successfully initialized Compact Memory memory container at {path}")


@memory_app.command(
    "stats",
    help="Displays statistics about the Compact Memory memory container.\n\nUsage Examples:\n  compact-memory memory stats\n  compact-memory memory stats --memory-path path/to/my_container --json",
)
def stats(
    json_output: bool = typer.Option(
        False, "--json", help="Output statistics in JSON format."
    ),
) -> None:
    container = load_memory_container(Path("."))
    data = container.get_statistics()
    logging.debug("Collected statistics: %s", data)
    if json_output:
        typer.echo(json.dumps(data))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


@memory_app.command(
    "validate", help="Validates the integrity of the memory container's storage."
)
def validate_memory_storage(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the memory container directory. Overrides global setting if provided.",
    ),
) -> None:
    final_memory_path_str = memory_path_arg
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


@memory_app.command(
    "clear",
    help="Deletes all data from a memory container. This action is irreversible.\n\nUsage Examples:\n  compact-memory memory clear --force\n  compact-memory memory clear --memory-path path/to/another_container --dry-run",
)
def clear(
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the memory container directory. Overrides global setting if provided.",
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
    final_memory_path_str = memory_path_arg
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
            f"Are you sure you want to delete all data in memory container at '{path}'? This cannot be undone.",
            abort=True,
        ):
            return  # Added path and warning
    if path.exists():
        shutil.rmtree(path)
        typer.echo(f"Successfully cleared memory container data at {path}")
    else:
        typer.secho(
            f"Error: Directory '{path}' not found.", err=True, fg=typer.colors.RED
        )  # Added path


# --- Top-Level Commands ---
@app.command("ingest", help="Legacy ingest command (no longer supported).")
def ingest() -> None:
    typer.secho("Ingestion functionality was removed.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.command(
    "query",
    help='Queries the Compact Memory memory container and returns an AI-generated response.\n\nUsage Examples:\n  compact-memory query "What is the capital of France?"\n  compact-memory query "Explain the theory of relativity in simple terms" --show-prompt-tokens',
)
def query(
    ctx: typer.Context,
    query_text: str = typer.Argument(
        ..., help="The query text to send to the memory container."
    ),
    show_prompt_tokens: bool = typer.Option(
        False,
        "--show-prompt-tokens",
        help="Display the token count of the final prompt sent to the LLM.",
    ),
) -> None:
    dim = get_embedding_dim()
    store = InMemoryVectorStore(embedding_dim=dim)
    container = MemoryContainer(store)
    final_model_id = ctx.obj.get("default_model_id")
    final_engine_id = ctx.obj.get("default_engine_id") # Updated variable name

    if final_model_id is None:
        typer.secho(
            "Error: Default Model ID not specified. Use --model-id option or set in config.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        container._chat_model = local_llm.LocalChatModel(model_name=final_model_id)
        container._chat_model.load_model()
    except RuntimeError as exc:
        typer.secho(
            f"Error loading chat model '{final_model_id}': {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)  # Added model_id to error

    mgr = ActiveMemoryManager()
    comp_engine_instance = None # Updated variable name
    if final_engine_id and final_engine_id.lower() != "none": # Updated variable name
        try:
            comp_cls = get_compression_engine(final_engine_id) # Updated function call and variable
            info = get_engine_metadata(final_engine_id) # Updated function call and variable
            if info and info.get("source") == "contrib":
                typer.secho(
                    "\u26a0\ufe0f Using experimental engine from contrib: not officially supported.", # Updated text
                    fg=typer.colors.YELLOW,
                )
            comp_engine_instance = comp_cls() # Updated variable name
        except KeyError:
            typer.secho(
                f"Error: Unknown compression engine '{final_engine_id}' (from global config/option). Available: {', '.join(available_engines())}", # Updated text and variables
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        try:
            result = container.receive_channel_message(
                "cli", query_text, mgr, compression=comp_engine_instance # Updated variable name
            )
        except TypeError:  # Older agents might not have compression param
            result = container.receive_channel_message("cli", query_text, mgr)

        reply = result.get("reply")
        if reply:
            typer.echo(reply)
        else:
            typer.secho(
                "The memory container did not return a reply.", fg=typer.colors.YELLOW
            )

        if show_prompt_tokens and result.get("prompt_tokens") is not None:
            typer.echo(f"Prompt tokens: {result['prompt_tokens']}")


@app.command(
    "compress",
    help='Compress text using a specified compression engine. Compresses text from a string, file, or directory.\n\nUsage Examples:\n  compact-memory compress --text "Some very long text..." --engine first_last_engine --budget 100\n  compact-memory compress --file path/to/document.txt -e prototype_engine -b 200 -o summary.txt\n  compact-memory compress --dir input_dir/ -e custom_package_engine -b 500 --output-dir output_dir/ --recursive -p "*.md"', # Updated examples
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
    engine_arg: Optional[str] = typer.Option( # Updated variable name
        None,
        "--engine", # Updated option
        "-e", # Updated short option
        help="Specify the CompressionEngine with --engine. Overrides the global default.", # Updated help
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output compressed result in JSON format to stdout",
    ),
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path of the memory container directory to store the compressed content",
    ),
    init_memory: bool = typer.Option(
        False,
        "--init-memory",
        help="Initialize a new memory container at --memory-path if it does not exist",
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
        help="Token budget for the compressed output. The engine will aim to keep the output within this limit.", # Updated help
    ),
    verbose_stats: bool = typer.Option(
        False,
        "--verbose-stats",
        help="Show detailed token counts and processing time per item.",
    ),
) -> None:
    final_engine_id = ( # Updated variable name
        engine_arg if engine_arg is not None else ctx.obj.get("default_engine_id") # Updated arg name and context key
    )
    if not final_engine_id: # Updated variable name
        typer.secho(
            "Error: Compression engine not specified. Use --engine option or set COMPACT_MEMORY_DEFAULT_ENGINE_ID / config.", # Updated text
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    final_memory_path_str = memory_path_arg

    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokenizer = lambda text, **k: {"input_ids": enc.encode(text)}
    except Exception:
        tokenizer = lambda t, **k: t.split()

    if sum(x is not None for x in (text, file, dir)) != 1:
        typer.echo("Error: specify exactly ONE of --text / --file / --dir", err=True)
        raise typer.Exit(1)

    if final_memory_path_str and (output_file or output_dir or json_output):
        typer.secho(
            "--memory-path cannot be combined with --output, --output-dir, or --json",
            err=True,
            fg=typer.colors.RED,
        )
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

    if final_memory_path_str:
        container = load_memory_container(Path(final_memory_path_str))
        if text is not None:
            if text == "-":
                text = sys.stdin.read()
            compress_text_to_memory(
                container,
                text,
                final_engine_id, # Updated variable
                budget,
                verbose_stats,
                tokenizer,
            )
        elif file is not None:
            compress_file_to_memory(
                container,
                file,
                final_engine_id, # Updated variable
                budget,
                verbose_stats,
                tokenizer,
            )
        else:
            compress_directory_to_memory(
                container,
                dir,
                final_engine_id, # Updated variable
                budget,
                recursive,
                pattern,
                verbose_stats,
                tokenizer,
            )
    else:
        if text is not None:
            if text == "-":
                text = sys.stdin.read()
            compress_text(
                text,
                final_engine_id, # Updated variable
                budget,
                output_file,
                output_trace,
                verbose_stats,
                tokenizer,
                json_output,
            )
        elif file is not None:
            compress_file(
                file,
                final_engine_id, # Updated variable
                budget,
                output_file,
                verbose_stats,
                tokenizer,
                output_trace,
                json_output,
            )
        else:
            if "**" in pattern and not recursive:
                typer.secho(
                    "Pattern includes '**' but --recursive not set; matching may miss subdirectories",
                    fg=typer.colors.YELLOW,
                )
            compress_directory(
                dir,
                final_engine_id, # Updated variable
                budget,
                output_dir,
                recursive,
                pattern,
                verbose_stats,
                tokenizer,
                None,
                json_output,
            )


# --- Helper functions for compress (formerly summarize/compress_text) ---
def compress_text(
    text_content: str,
    engine_id: str, # Updated parameter name
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    json_output: bool = False,
) -> None:
    try:
        engine_cls = get_compression_engine(engine_id) # Updated var and func names
        info = get_engine_metadata(engine_id) # Updated func name
        if info and info.get("source") == "contrib":
            typer.secho(
                "\u26a0\ufe0f Using experimental engine from contrib: not officially supported.", # Updated text
                fg=typer.colors.YELLOW,
            )
    except KeyError:
        typer.secho(
            f"Unknown compression engine '{engine_id}'. Available: {', '.join(available_engines())}", # Updated text and func name
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    engine = engine_cls() # Updated var name
    start = time.time()
    result = engine.compress(text_content, budget, tokenizer=tokenizer) # Updated var name
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
        if json_output:
            data = {
                "compressed": compressed.text,
                "original_tokens": token_count(tokenizer, text_content),
                "compressed_tokens": token_count(tokenizer, compressed.text),
            }
            typer.echo(json.dumps(data))
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


def compress_file(
    file_path: Path,
    engine_id: str, # Updated parameter name
    budget: int,
    output_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    trace_file: Optional[Path],
    json_output: bool,
) -> None:
    if not file_path.exists() or not file_path.is_file():
        typer.secho(f"File not found: {file_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        text_content = file_path.read_text()
    except (IOError, OSError, PermissionError) as exc:
        typer.secho(f"Error reading {file_path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    compress_text(
        text_content,
        engine_id, # Updated arg
        budget,
        output_file,
        trace_file,
        verbose_stats,
        tokenizer,
        json_output,
    )


def compress_directory(
    dir_path: Path,
    engine_id: str, # Updated parameter name
    budget: int,
    output_dir_param: Optional[Path],
    recursive: bool,
    pattern: str,
    verbose_stats: bool,
    tokenizer: Any,
    trace_file: Optional[Path],
    json_output: bool,
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
        compress_file(
            input_file,
            engine_id, # Updated arg
            budget,
            out_path,
            verbose_stats,
            tokenizer,
            None,
            json_output,
        )
        count += 1
    if verbose_stats:
        typer.echo(f"Processed {count} files.")


def compress_text_to_memory(
    container: MemoryContainer,
    text: str,
    engine_id: str, # Updated parameter name
    budget: int,
    verbose_stats: bool,
    tokenizer: Any,
    source_document_id: str | None = None,
) -> None:
    start = time.time()
    try:
        engine_cls = get_compression_engine(engine_id) # Updated var and func names
    except KeyError:
        typer.secho(
            f"Unknown compression engine '{engine_id}'. Available: {', '.join(available_engines())}", # Updated text and func name
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    engine = engine_cls() # Updated var name
    result = engine.compress(text, budget, tokenizer=tokenizer) # Updated var name
    if isinstance(result, tuple):
        compressed, _ = result
    else:
        compressed = result
    container.add_memory(compressed.text, source_document_id=source_document_id)
    elapsed = (time.time() - start) * 1000
    if verbose_stats:
        orig_tokens = token_count(tokenizer, text)
        comp_tokens = token_count(tokenizer, compressed.text)
        typer.echo(
            f"Original tokens: {orig_tokens}\nCompressed tokens: {comp_tokens}\nTime ms: {elapsed:.1f}"
        )


def compress_file_to_memory(
    container: MemoryContainer,
    file_path: Path,
    engine_id: str, # Updated parameter name
    budget: int,
    verbose_stats: bool,
    tokenizer: Any,
) -> None:
    if not file_path.exists() or not file_path.is_file():
        typer.secho(f"File not found: {file_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    text_content = file_path.read_text()
    compress_text_to_memory(
        container,
        text_content,
        engine_id, # Updated arg
        budget,
        verbose_stats,
        tokenizer,
        source_document_id=str(file_path),
    )


def compress_directory_to_memory(
    container: MemoryContainer,
    dir_path: Path,
    engine_id: str, # Updated parameter name
    budget: int,
    recursive: bool,
    pattern: str,
    verbose_stats: bool,
    tokenizer: Any,
) -> None:
    if not dir_path.exists() or not dir_path.is_dir():
        typer.secho(f"Directory not found: {dir_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    files = list(dir_path.rglob(pattern) if recursive else dir_path.glob(pattern))
    if not files:
        typer.echo("No matching files found.")
        return
    for input_file in files:
        typer.echo(f"Processing {input_file}...")
        compress_file_to_memory(
            container,
            input_file,
            engine_id, # Updated arg
            budget,
            verbose_stats,
            tokenizer,
        )


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
    "list-engines", # Updated command name
    help="Lists all available compression engine IDs, their versions, and sources (built-in or plugin).", # Updated help
)
def list_engines( # Updated function name
    include_contrib: bool = typer.Option(
        False, "--include-contrib", help="Include experimental contrib engines." # Updated help
    )
) -> None:
    """Displays a table of registered compression engines.""" # Updated docstring
    load_plugins()  # Ensure plugins are loaded
    enable_all_experimental_engines() # Updated function name
    # Experimental engines register themselves; no legacy contrib layer
    table = Table(
        "Engine ID", # Updated header
        "Display Name",
        "Version",
        "Source",
        "Status",
        title="Available CompressionEngines", # Updated title
    )
    meta = all_engine_metadata() # Updated function name
    ids = available_engines() # Updated function name
    if not ids:
        typer.echo("No compression engines found.") # Updated text
        return
    for eid in sorted(ids): # Updated var name
        info = meta.get(eid, {}) # Updated var name
        if not include_contrib and info.get("source") == "contrib":
            continue
        status = ""
        if info.get("overrides"):
            status = f"Overrides '{info['overrides']}'"
        table.add_row(
            eid, # Updated var name
            info.get("display_name", eid) or eid, # Updated var name
            info.get("version", "N/A") or "N/A",
            info.get("source", "built-in") or "built-in",
            status,
        )
    console.print(table)


@dev_app.command(
    "inspect-engine", # Updated command name
    help="Inspects aspects of a compression engine, currently focused on 'prototype_engine' engine's beliefs.", # Updated help
)
def inspect_engine( # Updated function name
    engine_name: str = typer.Argument( # Updated argument name
        ...,
        help="The name of the engine to inspect. Currently, only 'prototype_engine' is supported.", # Updated help
    ),
    *,
    list_prototypes: bool = typer.Option(
        False,
        "--list-prototypes",
        help="List consolidated prototypes (beliefs) if the engine is 'prototype_engine' and a memory container path is provided.", # Updated help
    ),
) -> None:
    if engine_name.lower() != "prototype_engine": # Updated var and value
        typer.secho(
            f"Error: Inspection for engine '{engine_name}' is not supported. Only 'prototype_engine' is currently inspectable.", # Updated text and var
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if list_prototypes:
        dim = get_embedding_dim()
        store = InMemoryVectorStore(embedding_dim=dim)
        container = MemoryContainer(store)
        protos = container.get_prototypes_view()
        if not protos:
            typer.echo("No prototypes found.")
            return
        table = Table(
            "ID",
            "Strength",
            "Confidence",
            "Summary",
            title="Prototypes for MemoryContainer",
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
            f"Engine '{engine_name}' is available. Use --list-prototypes and provide a memory container path to see its beliefs." # Updated text and var
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
    "test-llm-prompt",
    help='Tests a Language Model (LLM) prompt with specified context and query.\n\nUsage Examples:\n  compact-memory dev test-llm-prompt --context "AI is rapidly evolving." --query "Tell me more." --model-id tiny-gpt2\n  cat context.txt | compact-memory dev test-llm-prompt --context - -q "What are the implications?" --model-id openai/gpt-3.5-turbo --output-response response.txt --llm-config my_llm_config.yaml',
)
def test_llm_prompt(
    *,
    context_input: str = typer.Option(
        ...,
        "--context",
        "-c",
        help="Context string for the LLM, path to a context file, or '-' to read from stdin.",
    ),
    query: str = typer.Option(
        ..., "--query", "-q", help="User query to append to the context for the LLM."
    ),
    model_id: str = typer.Option(
        "tiny-gpt2",
        "--model",
        help="Model ID to use for the test (must be defined in LLM config).",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt",
        "-s",
        help="Optional system prompt to prepend to the main prompt.",
    ),
    max_new_tokens: int = typer.Option(
        150, help="Maximum number of new tokens the LLM should generate."
    ),
    output_llm_response_file: Optional[Path] = typer.Option(
        None,
        "--output-response",
        help="File path to save the LLM's raw response. If unspecified, prints to console.",
    ),
    llm_config_file: Optional[Path] = typer.Option(
        Path("llm_models_config.yaml"),
        "--llm-config",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the LLM configuration YAML file.",
    ),
    api_key_env_var: Optional[str] = typer.Option(
        None,
        help="Environment variable name that holds the API key for the LLM provider (e.g., 'OPENAI_API_KEY').",
    ),
) -> None:
    def read_val(
        val: str, input_name: str
    ) -> str:  # Added input_name for better error messages
        if val == "-":
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

    context_text = read_val(context_input, "context")

    cfg = {}
    if (
        llm_config_file and llm_config_file.exists()
    ):  # Check existence again as Typer's check might be bypassed in some complex setups
        try:
            cfg = yaml.safe_load(llm_config_file.read_text()) or {}
        except Exception as exc:
            typer.secho(
                f"Error loading LLM config '{llm_config_file}': {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    model_cfg = cfg.get(model_id, {"provider": "local", "model_name": model_id})
    provider_name = model_cfg.get("provider")
    actual_model_name = model_cfg.get(
        "model_name", model_id
    )  # Use actual_model_name to avoid conflict

    if provider_name == "openai":
        provider = llm_providers.OpenAIProvider()
    elif provider_name == "gemini":
        provider = llm_providers.GeminiProvider()
    else:
        provider = (
            llm_providers.LocalTransformersProvider()
        )  # Default or if provider_name is 'local'

    api_key = os.getenv(api_key_env_var) if api_key_env_var else None

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    if context_text:
        prompt_parts.append(
            context_text
        )  # Ensure context_text is not empty before adding
    prompt_parts.append(query)
    prompt = "\n\n".join(
        part for part in prompt_parts if part
    )  # Use double newline for better separation

    typer.echo(f"--- Sending Prompt to LLM ({provider_name} - {actual_model_name}) ---")
    typer.echo(
        prompt[:500] + "..." if len(prompt) > 500 else prompt
    )  # Preview long prompts
    typer.echo("--- End of Prompt ---")

    try:
        response = provider.generate_response(
            prompt,
            model_name=actual_model_name,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
        )
    except Exception as exc:
        typer.secho(f"LLM generation error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if output_llm_response_file:
        try:
            output_llm_response_file.parent.mkdir(parents=True, exist_ok=True)
            output_llm_response_file.write_text(response)
            typer.echo(f"LLM response saved to: {output_llm_response_file}")
        except Exception as exc:
            typer.secho(
                f"Error writing LLM response to '{output_llm_response_file}': {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    else:
        typer.echo("\n--- LLM Response ---")
        typer.echo(response)


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
        "compact_memory_example_strategy",
        "--name",
        help="Name for the new strategy package (e.g., 'compact_memory_my_strategy'). Used for directory and strategy ID.",
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

    engine_py_content = f"""from CompressionEngine.core.engines_abc import CompressionEngine, CompressedMemory, CompressionTrace # Updated import
# Add any other necessary imports here

class MyEngine(CompressionEngine): # Updated class name
    # Unique identifier for your engine
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
            engine_name=self.id, # Updated param
            # original_tokens=len(tokenizer(str(text_or_chunks))['input_ids']) if tokenizer else None, # Example, ensure tokenizer call is correct
            # compressed_tokens=len(tokenizer(compressed_text)['input_ids']) if tokenizer else None, # Example
            engine_params={{"llm_token_budget": llm_token_budget}}, # Example
            input_summary={{"original_length": len(str(text_or_chunks))}}, # Example
            output_summary={{"compressed_length": len(compressed_text)}}, # Example
            # Add more trace details as needed
        )
        return CompressedMemory(text=compressed_text), trace
"""
    (target_dir / "engine.py").write_text(engine_py_content) # Updated filename

    manifest = {
        "package_format_version": "1.0",
        "engine_id": name, # Updated key
        "engine_class_name": "MyEngine", # Updated key
        "engine_module": "engine", # Updated key and value
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
    if data.get("engine_name"): # Updated key
        title += f" (Engine: {data['engine_name']})" # Updated text and key
    if data.get("original_tokens"):
        title += f" | Original Tokens: {data['original_tokens']}"
    if data.get("compressed_tokens"):
        title += f" | Compressed Tokens: {data['compressed_tokens']}"
    if data.get("processing_ms"):
        title += f" | Time: {data['processing_ms']:.2f}ms"

    console.print(f"Engine: {data.get('engine_name', '')}") # Updated text and key
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
        help=f"The configuration key to set. Valid keys: {', '.join(DEFAULT_CONFIG.keys())}.", # This help might need update if keys change e.g. default_strategy_id
    ),
    value: str = typer.Argument(..., help="The new value for the configuration key."),
) -> None:
    config: Config = ctx.obj["config"]
    try:
        # The key itself (e.g. "default_strategy_id") would need to be changed in Config class for this to fully work
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
        help=f"Specific configuration key to display. Valid keys: {', '.join(DEFAULT_CONFIG.keys())}.", # May need update if keys change
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

# --- Dev App Commands for package creation ---
# Need to update create-strategy-package and validate-strategy-package to engine equivalents.
# The following are placeholders for where the original dev_app.command decorators were implicitly.

@dev_app.command(
    "create-engine-package", # Updated command
    help="Creates a new compression engine extension package from a template. This command generates a template directory with all the necessary files to start developing a new, shareable engine package, including a sample engine, manifest file, and README.", # Updated help
)
def create_engine_package( # Updated function name
    name: str = typer.Option(
        "compact_memory_example_engine", # Updated default name
        "--name",
        help="Name for the new engine package (e.g., 'compact_memory_my_engine'). Used for directory and engine ID.", # Updated help
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        help="Directory where the engine package will be created. Defaults to a new directory named after the engine in the current location.", # Updated help
    ),
) -> None:
    target_dir = Path(path or name).resolve()

    if target_dir.exists() and any(target_dir.iterdir()):
        typer.secho(
            f"Error: Output directory '{target_dir}' already exists and is not empty.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "experiments").mkdir(exist_ok=True)

    engine_py_content = f"""from CompressionEngine.core.engines_abc import CompressionEngine, CompressedMemory, CompressionTrace # Updated import
# Add any other necessary imports here

class MyEngine(CompressionEngine): # Updated class name
    # Unique identifier for your engine
    id = "{name}"

    # Optional: Define parameters your engine accepts with default values
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
            engine_name=self.id, # Updated param
            # original_tokens=len(tokenizer(str(text_or_chunks))['input_ids']) if tokenizer else None, # Example, ensure tokenizer call is correct
            # compressed_tokens=len(tokenizer(compressed_text)['input_ids']) if tokenizer else None, # Example
            engine_params={{"llm_token_budget": llm_token_budget}}, # Example
            input_summary={{"original_length": len(str(text_or_chunks))}}, # Example
            output_summary={{"compressed_length": len(compressed_text)}}, # Example
            # Add more trace details as needed
        )
        return CompressedMemory(text=compressed_text), trace
"""
    (target_dir / "engine.py").write_text(engine_py_content) # Updated filename

    manifest = {
        "package_format_version": "1.0",
        "engine_id": name, # Updated key
        "engine_class_name": "MyEngine", # Updated key
        "engine_module": "engine", # Updated key and value
        "display_name": name,
        "version": "0.1.0",
        "authors": [],
        "description": "Describe the engine", # Updated text
    }
    (target_dir / "engine_package.yaml").write_text(yaml.safe_dump(manifest)) # Updated filename
    (target_dir / "requirements.txt").write_text("\n")
    (target_dir / "README.md").write_text(f"# {name}\n")
    (target_dir / "experiments" / "example.yaml").write_text(
        """dataset: example.txt\nparam_grid:\n- {}\npackaged_engine_config:\n  engine_params: {}\n""" # Updated key
    )
    typer.echo(f"Successfully created engine package '{name}' at: {target_dir}") # Updated text


@dev_app.command(
    "validate-engine-package", # Updated command
    help="Validates the structure and manifest of a compression engine extension package.\n\nUsage Examples:\n  compact-memory dev validate-engine-package path/to/my_engine_pkg", # Updated help
)
def validate_engine_package( # Updated function name
    package_path: Path = typer.Argument(
        ...,
        help="Path to the root directory of the engine package.", # Updated help
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    )
) -> None:
    errors, warnings = validate_package_dir(package_path) # validate_package_dir will need internal changes
    for w in warnings:
        typer.secho(f"Warning: {w}", fg=typer.colors.YELLOW)
    if errors:
        for e in errors:
            typer.echo(e)
        raise typer.Exit(code=1)
    typer.echo(f"Engine package at '{package_path}' appears valid.") # Updated text


if __name__ == "__main__":
    app()
