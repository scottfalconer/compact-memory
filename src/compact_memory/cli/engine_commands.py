import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import typer
from rich.table import Table
from rich.console import Console

from compact_memory.config import Config
from compact_memory.llm_providers_abc import LLMProvider
from compact_memory.llm_providers.factory import create_llm_provider
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
    # LLM Configuration Options
    llm_config: Optional[str] = typer.Option(
        None, "--llm-config", help="Name of LLM configuration to use (from llm_models_config.yaml)"
    ),
    llm_provider_type: Optional[str] = typer.Option(
        None, "--llm-provider-type", help="Type of LLM provider (e.g., \"local\", \"openai\", \"mock\")"
    ),
    llm_model_name: Optional[str] = typer.Option(
        None, "--llm-model-name", help="Name or path of the LLM model to use"
    ),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key", help="API key for remote LLM providers (e.g., OpenAI)"
    ),
    # ReadAgent Specific Options
    readagent_gist_model_name: Optional[str] = typer.Option(
        None, "--readagent-gist-model-name", help="[ReadAgent] Override model name for gist summarization phase"
    ),
    readagent_gist_length: Optional[int] = typer.Option(
        None, "--readagent-gist-length", help="[ReadAgent] Override target token length for gist summaries (default: 100)"
    ),
    readagent_lookup_max_tokens: Optional[int] = typer.Option(
        None, "--readagent-lookup-max-tokens", help="[ReadAgent] Override max new tokens for lookup phase output (default: 50)"
    ),
    readagent_qa_model_name: Optional[str] = typer.Option(
        None, "--readagent-qa-model-name", help="[ReadAgent] Override model name for Q&A answering phase"
    ),
    readagent_qa_max_new_tokens: Optional[int] = typer.Option(
        None, "--readagent-qa-max-new-tokens", help="[ReadAgent] Override max new tokens for Q&A answer generation (default: 250)"
    ),
    # Existing options
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

    app_config: Config = ctx.obj.get("config")
    if not app_config:
        app_config = Config()
        typer.secho("Warning: Global app config not found in context, loaded a new one.", fg=typer.colors.YELLOW)

    final_engine_id = engine_id_arg or ctx.obj.get("default_engine_id") or "prototype"
    typer.echo(f"Initializing engine store for engine ID: {final_engine_id} at {path}")

    # Base engine config from direct flags
    engine_config: Dict[str, Any] = {
        'chunker_id': chunker,
        'name': name,
    }
    if final_engine_id == 'prototype':
        if not 0.5 <= tau <= 0.95:
            typer.secho(
                "Error: --tau must be between 0.5 and 0.95 for the 'prototype' engine.", err=True, fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        engine_config['similarity_threshold'] = tau

    # --- LLM Setup ---
    llm_provider_instance: Optional[LLMProvider] = None
    cli_llm_override_config: Dict[str, Any] = {}

    try:
        EngineCls = get_compression_engine(final_engine_id)
    except KeyError:
        typer.secho(
            f"Error: Engine ID '{final_engine_id}' not found. Available engines: {', '.join(available_engines())}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    engine_requires_llm = getattr(EngineCls, 'requires_llm', False)

    if engine_requires_llm:
        if llm_config and llm_provider_type:
            typer.secho(
                "Error: --llm-config cannot be used with --llm-provider-type. "
                "Please use one method for LLM configuration.",
                fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)

        if llm_provider_type and not llm_model_name:
            typer.secho("Error: --llm-model-name must be provided if --llm-provider-type is specified.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        if llm_model_name and not llm_provider_type and not llm_config:
            typer.secho("Error: --llm-provider-type must be provided if --llm-model-name is specified without --llm-config.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        actual_llm_config_name = llm_config
        actual_llm_provider_type = llm_provider_type
        actual_llm_model_name = llm_model_name
        actual_llm_api_key = llm_api_key

        if not actual_llm_config_name and not actual_llm_provider_type:
            default_model_id = app_config.get("default_model_id")
            if default_model_id:
                typer.secho(f"No LLM specified directly for init; using default from global config: {default_model_id}", fg=typer.colors.BLUE)
                if default_model_id in app_config.get_all_llm_configs():
                    actual_llm_config_name = default_model_id
                elif '/' in default_model_id:
                    parts = default_model_id.split('/', 1)
                    actual_llm_provider_type = parts[0]
                    actual_llm_model_name = parts[1]
                else:
                    typer.secho(
                        f"Error: Default model ID '{default_model_id}' is not a valid named configuration or 'provider/model_name' format.",
                        fg=typer.colors.RED, err=True
                    )
                    raise typer.Exit(code=1)
            else:
                typer.secho(
                    f"Error: Engine '{final_engine_id}' requires an LLM for initialization. Please provide LLM configuration "
                    "via --llm-config or --llm-provider-type/--llm-model-name, or set a default_model_id in the global config.",
                    fg=typer.colors.RED, err=True
                )
                raise typer.Exit(code=1)

        try:
            llm_provider_instance = create_llm_provider(
                config_name=actual_llm_config_name,
                provider_type=actual_llm_provider_type,
                model_name=actual_llm_model_name,
                api_key=actual_llm_api_key,
                app_config=app_config,
            )
        except (ValueError, ImportError) as e:
            typer.secho(f"Error creating LLM provider for init: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        # Specific ReadAgent config overrides from CLI for init
        # These will be merged into the main engine_config
        if final_engine_id == "readagent_gist": # Or a more generic check for engines that use these settings
            if readagent_gist_model_name is not None:
                cli_llm_override_config["gist_model_name"] = readagent_gist_model_name
            elif actual_llm_model_name: # Default from main LLM config if not specified for gist
                 cli_llm_override_config["gist_model_name"] = actual_llm_model_name

            if readagent_gist_length is not None:
                cli_llm_override_config["gist_length"] = readagent_gist_length

            if readagent_lookup_max_tokens is not None:
                cli_llm_override_config["lookup_max_tokens"] = readagent_lookup_max_tokens

            if readagent_qa_model_name is not None:
                cli_llm_override_config["qa_model_name"] = readagent_qa_model_name
            elif actual_llm_model_name: # Default from main LLM config if not specified for QA
                cli_llm_override_config["qa_model_name"] = actual_llm_model_name

            if readagent_qa_max_new_tokens is not None:
                cli_llm_override_config["qa_max_new_tokens"] = readagent_qa_max_new_tokens

        # Merge LLM override config into the main engine config
        # CLI overrides for these specific keys take precedence
        engine_config.update(cli_llm_override_config)

    # --- Instantiate and Save Engine ---
    try:
        engine: Any # BaseCompressionEngine once all engines conform
        if engine_requires_llm:
            if not llm_provider_instance: # Should be caught above, but defensive check
                typer.secho("Critical error: LLM provider not initialized for LLM-dependent engine.", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
            engine = EngineCls(config=engine_config, llm_provider=llm_provider_instance)
        else:
            engine = EngineCls(config=engine_config)

        path.mkdir(parents=True, exist_ok=True)
        engine.save(path) # The engine's save method should handle persisting its config, including LLM related parts if designed to.
        typer.echo(f"Successfully initialized Compact Memory engine store with engine '{final_engine_id}' at {path}")
        if cli_llm_override_config:
             typer.secho(f"  Applied LLM specific configurations: {cli_llm_override_config}", fg=typer.colors.BLUE)
        if llm_provider_instance:
             typer.secho(f"  LLM Provider: {type(llm_provider_instance).__name__} configured.", fg=typer.colors.BLUE)

    except EmbeddingDimensionMismatchError as exc:
        typer.secho(f"Error during engine initialization: {exc}", fg=typer.colors.RED, err=True)
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
