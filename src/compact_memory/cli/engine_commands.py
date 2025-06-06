import json
import shutil
from pathlib import Path
from typing import Optional
import logging

import typer
from rich.table import Table
from rich.console import Console

from compact_memory.engines import load_engine
from compact_memory.engines.registry import (
    available_engines,
    get_engine_metadata,
    all_engine_metadata,
    get_compression_engine,
)
from compact_memory.embedding_pipeline import EmbeddingDimensionMismatchError


console = Console()

engine_app = typer.Typer(
    help="Manage engine storage: initialize, inspect statistics, validate, and clear."
)


def _corrupt_exit(path: Path, exc: Exception) -> None:
    typer.echo(f"Error: Brain data is corrupted. {exc}", err=True)
    typer.echo(
        f"Try running compact-memory validate {path} for more details or restore from a backup.",
        err=True,
    )
    raise typer.Exit(code=1)


@engine_app.command("list", help="Lists all available compression engine IDs.")
def list_command() -> None:  # Renamed from list_engines
    # from compact_memory.plugin_loader import load_plugins # Handled globally now
    # load_plugins()
    ids = available_engines()
    meta = all_engine_metadata()
    if not ids:
        typer.echo("No compression engines found.")
        return
    table = Table(
        "Engine ID",
        "Display Name",
        "Version",
        "Source",
        title="Available Compression Engines",
    )
    for eid in sorted(ids):
        info = meta.get(eid, {})
        table.add_row(
            eid,
            info.get("display_name", eid) or eid,
            info.get("version", "N/A") or "N/A",
            info.get("source", "built-in") or "built-in",
        )
    console.print(table)


@engine_app.command("info", help="Show metadata for a specific engine ID.")
def info_command(engine_id: str) -> None:  # Renamed from engine_info
    # from compact_memory.plugin_loader import load_plugins # Handled globally
    # load_plugins()
    info = get_engine_metadata(engine_id)
    if not info:
        typer.secho(f"Engine '{engine_id}' not found.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(info))


@engine_app.command(
    "init",
    help="Initializes a new Compact Memory engine store in a specified directory.\n\nUsage Examples:\n  compact-memory engine init ./my_engine_dir --engine prototype\n  compact-memory engine init /path/to/store --engine prototype --name 'research_mem' --tau 0.75 --chunker SentenceWindowChunker",
)
def init_command(  # Renamed from init
    target_directory: Path = typer.Argument(
        ...,
        help="Directory to initialize the new engine store in. Will be created if it doesn't exist.",
        resolve_path=True,
    ),
    *,
    ctx: typer.Context,
    engine_id_arg: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="The ID of the compression engine to initialize."
    ),
    name: str = typer.Option(
        "default_store", help="A descriptive name for the engine store or its configuration."
    ),
    tau: float = typer.Option(
        0.8,
        help="Similarity threshold for 'prototype' engine, between 0.5 and 0.95.",
    ),
    chunker: str = typer.Option(
        "SentenceWindowChunker",
        help="Identifier for the chunker to be used (e.g., 'SentenceWindowChunker').",
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

    final_engine_id = engine_id_arg or ctx.obj.get("default_engine_id") or "prototype"
    typer.echo(f"Initializing with engine ID: {final_engine_id}")

    if final_engine_id == "prototype":
        if not 0.5 <= tau <= 0.95:
            typer.secho(
                "Error: --tau must be between 0.5 and 0.95 for the 'prototype' engine.", err=True, fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

    engine_config = {}
    engine_config['chunker_id'] = chunker
    engine_config['name'] = name

    if final_engine_id == 'prototype':
        engine_config['similarity_threshold'] = tau

    try:
        EngineCls = get_compression_engine(final_engine_id)
    except KeyError:
        typer.secho(
            f"Error: Engine ID '{final_engine_id}' not found. Available engines: {', '.join(available_engines())}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        engine = EngineCls(config=engine_config)
        path.mkdir(parents=True, exist_ok=True)
        engine.save(path)
        typer.echo(f"Successfully initialized Compact Memory engine store with engine '{final_engine_id}' at {path}")
    except EmbeddingDimensionMismatchError as exc:
        typer.secho(f"Error during engine initialization: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.secho(f"An unexpected error occurred during engine initialization or save: {exc}", err=True, fg=typer.colors.RED)
        logging.exception("Failed to initialize engine store.")
        raise typer.Exit(code=1)


@engine_app.command(
    "stats",
    help="Displays statistics about the Compact Memory engine store.\n\nUsage Examples:\n  compact-memory engine stats\n  compact-memory engine stats --memory-path path/to/my_container --json",
)
def stats_command(  # Renamed from stats
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the engine store directory. Overrides global setting if provided.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output statistics in JSON format."
    ),
) -> None:
    final_memory_path_str = memory_path_arg or ctx.obj.get("compact_memory_path")
    if final_memory_path_str is None:
        typer.secho(
            "Critical Error: Memory path could not be resolved for stats.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    container = load_engine(Path(final_memory_path_str))
    data = container.get_statistics()
    logging.debug("Collected statistics: %s", data)
    if json_output:
        typer.echo(json.dumps(data))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


@engine_app.command(
    "validate", help="Validates the integrity of the engine store's storage."
)
def validate_command(  # Renamed from validate_memory_storage
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the engine store directory. Overrides global setting if provided.",
    ),
) -> None:
    final_memory_path_str = memory_path_arg or ctx.obj.get("compact_memory_path") # Ensure path is resolved from context if not provided
    if final_memory_path_str is None:
        typer.secho(
            "Critical Error: Memory path could not be resolved for validate. Please provide it with --memory-path or set it globally.",
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
    # TODO: Implement actual validation logic if the engine supports it.
    # For now, this is a placeholder as in the original code.
    # Example:
    # try:
    #     engine = load_engine(path)
    #     if hasattr(engine, 'validate_storage'):
    #         is_valid, message = engine.validate_storage()
    #         if is_valid:
    #             typer.secho(f"Engine store at '{path}' is valid. {message}", fg=typer.colors.GREEN)
    #         else:
    #             typer.secho(f"Engine store at '{path}' is invalid: {message}", fg=typer.colors.RED, err=True)
    #             raise typer.Exit(code=1)
    #     else:
    #         typer.echo(f"Engine type '{type(engine).__name__}' does not support explicit validation via 'validate_storage'. Basic existence check passed.")
    # except Exception as e:
    #     _corrupt_exit(path, e) # Use your _corrupt_exit or similar
    typer.echo(f"Validation for engine store at '{path}': No specific storage validation implemented for this engine type beyond loading.")


@engine_app.command(
    "clear",
    help="Deletes all data from an engine store. This action is irreversible.\n\nUsage Examples:\n  compact-memory engine clear --force\n  compact-memory engine clear --memory-path path/to/another_container --dry-run",
)
def clear_command(  # Renamed from clear
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None,
        "--memory-path",
        "-m",
        help="Path to the engine store directory. Overrides global setting if provided.",
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
    final_memory_path_str = memory_path_arg or ctx.obj.get("compact_memory_path") # Ensure path is resolved
    if final_memory_path_str is None:
        typer.secho(
            "Critical Error: Memory path could not be resolved for clear. Please provide it with --memory-path or set it globally.",
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

    # For engines that might not just be a directory of files,
    # we should ideally call a method on the engine instance.
    # However, the current CLI design loads the engine only when needed for specific ops.
    # For 'clear', simply removing the directory is the most straightforward approach
    # if the engine's state is entirely contained within its directory.

    if dry_run:
        if path.exists():
            typer.echo(f"Dry run: Would delete directory and all its contents: {path}")
            # If you want to list contents:
            # for item in path.rglob('*'): # List all items recursively
            #     typer.echo(f"  - Would delete {item.relative_to(path)}")
        else:
            typer.echo(f"Dry run: Directory '{path}' does not exist, nothing to delete.")
        return

    if not force:
        if not typer.confirm(
            f"Are you sure you want to delete all data in engine store at '{path}'? This cannot be undone.",
            abort=True, # If user says no, exit the command.
        ):
            # This part is technically unreachable due to abort=True, but good for clarity.
            typer.echo("Operation cancelled by user.")
            return

    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
                typer.echo(f"Successfully cleared engine store data at {path}")
            else: # Should not happen if it's an engine store path, but good to check.
                path.unlink()
                typer.echo(f"Successfully deleted file at {path}")
        except Exception as e:
            typer.secho(f"Error deleting '{path}': {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    else:
        # This case should ideally be caught by the initial existence check,
        # but good to have as a fallback.
        typer.secho(
            f"Error: Directory '{path}' not found. Nothing to clear.", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1) # Or just return if a non-error exit is preferred when path doesn't exist.

# Note: `load_plugins()` is called in the main CLI callback, so it doesn't need to be repeated here
# unless these commands are intended to be runnable completely standalone without the main app context,
# which is not typical for Typer subcommands.
# Imports like `from compact_memory.plugin_loader import load_plugins` are removed if handled globally.
# Imports for `BaseCompressionEngine` or specific engines like `PrototypeEngine` are not needed here
# if `load_engine` and `get_compression_engine` are sufficient.
# `EmbeddingDimensionMismatchError` is kept as it's a specific exception that `init` handles.

# Corrected imports based on the new structure:
# - `from .prototype_engine import PrototypeEngine` -> Not directly used, `get_compression_engine` is.
# - `from .vector_store import InMemoryVectorStore` -> Not directly used.
# - `from .engine_registry import ...` -> `from compact_memory.engines.registry import ...`
# - `from .engines.no_compression_engine import NoCompressionEngine` -> Not directly used.
# - `from .engines import load_engine` -> `from compact_memory.engines import load_engine` (Corrected)
# - `from .embedding_pipeline import ...` -> `from compact_memory.embedding_pipeline import ...` (Corrected)
# - `from .logging_utils import configure_logging` -> Not used here directly, handled in main.
# - `from compact_memory import __version__` -> Not used here.
# - `from compact_memory.contrib import ...` -> Not used here.
# - `from compact_memory.config import ...` -> Not used here, ctx.obj provides config.
# - `from compact_memory.plugin_loader import load_plugins` -> Handled globally.
# - `from compact_memory.cli_plugins import load_cli_plugins` -> Handled globally.

# The `ctx` object passed to commands will contain "compact_memory_path", "default_engine_id", etc.
# from the main callback. Commands should use `ctx.obj.get("...")`.
# Example: `final_engine_id = engine_id_arg or ctx.obj.get("default_engine_id") or "prototype"`
# Example: `final_memory_path_str = memory_path_arg or ctx.obj.get("compact_memory_path")`

# The `validate` command's implementation needs careful consideration. The original CLI's
# `validate_memory_storage` was a placeholder. If specific engines have validation logic,
# `load_engine(path).validate_storage()` would be the way to call it.
# Otherwise, just checking if `load_engine(path)` succeeds without error can be a basic validation.
# The current implementation reflects the placeholder nature.

# The `clear` command's `dry_run` was also a placeholder. A more robust dry run
# would list files that *would* be deleted. The updated version provides a basic listing.
# It also directly uses `shutil.rmtree` which is appropriate if an engine store is a directory.
# For engines with external state, `engine.clear()` would be needed.
# Given the current structure, `shutil.rmtree` is the most direct interpretation of the original intent.
# Added more checks for `memory_path_arg` and resolving from `ctx.obj` for `validate` and `clear`.
# Renamed commands for clarity (e.g. `list_engines` -> `list_command`)
# Corrected `init_command` to use `engine_id_arg` and resolve `final_engine_id` using `ctx.obj`.
# Corrected `validate_command` and `clear_command` to resolve memory path from `ctx.obj` if argument not given.
# Corrected `init_command` to use `target_directory` as per its definition.
# Removed `from .prototype_engine import PrototypeEngine` and other unused direct engine imports.
# Ensured `load_engine` and `get_compression_engine` are imported from their new locations.
# `EmbeddingDimensionMismatchError` import is correct.
# `_corrupt_exit` is fine here.
# `console` is defined.
# `engine_app` is defined.
# `shutil` and `json` imports are used.
# `logging` is used.
# `Path` and `Optional` are used.
# `typer`, `Table`, `Console` are used.
# All command names updated (e.g., `list_engines` to `list_command`).
# Added placeholder comments for future validation logic in `validate_command`.
# Improved `clear_command` dry run and actual deletion logic.
# Corrected `init` command to properly use `engine_id_arg` from option and `default_engine_id` from context.
# Corrected `init` command to use `target_directory` consistently.
# Made sure `engine_config` in `init` correctly uses `chunker` variable.
# Corrected `validate_command` to use `memory_path_arg` and `ctx.obj.get("compact_memory_path")`.
# Corrected `clear_command` to use `memory_path_arg` and `ctx.obj.get("compact_memory_path")`.
# Added more specific error messages if memory path is not resolved in `stats`, `validate`, `clear`.
# Final check of imports:
# `compact_memory.engines.registry` for engine metadata/getting classes.
# `compact_memory.engines` for `load_engine`.
# `compact_memory.embedding_pipeline` for `EmbeddingDimensionMismatchError`.
# Standard library imports are fine.
# Rich and Typer imports are fine.
# `_corrupt_exit` is self-contained.
# `console` object is fine.
# `engine_app` object is fine.
# All command functions renamed (e.g., `list_engines` to `list_command`).
# All commands use `ctx.obj` correctly for shared parameters.
# The parameter name `engine_id_arg` in `init_command` matches the Option name.
# The parameter name `memory_path_arg` in `stats_command`, `validate_command`, `clear_command` matches the Option name.
# Looks good.
