import typer
from pathlib import Path
from typing import Optional
import logging
import sys  # For sys.stdin.isatty()

from compact_memory import __version__
from compact_memory.logging_utils import configure_logging  # Updated import
from compact_memory.config import Config  # Updated import
from compact_memory.plugin_loader import load_plugins  # Updated import
from compact_memory.contrib import enable_all_experimental_engines  # Updated import

# Import command groups and individual commands from their new modules
from .engine_commands import engine_app
from .config_commands import config_app
from .dev_commands import dev_app
from .query_commands import query_command  # Assuming it's named query_command now
from .compress_commands import (
    compress_command,
)  # Assuming it's named compress_command now

# --- Main Application ---
app = typer.Typer(
    help="Compact Memory: manage engine stores and advanced compression. Query and compress information. Manage engine store configurations and developer tools."
)

# --- Add Command Groups and Standalone Commands to Main App ---
app.add_typer(engine_app, name="engine")
app.add_typer(config_app, name="config")
app.add_typer(dev_app, name="dev")

# Add standalone commands directly
# The function itself is passed to app.command(), Typer handles the rest.
# The name in @app.command("name") will be how it's called in the CLI.
app.command("query")(query_command)
app.command("compress")(compress_command)


# --- Global Callbacks and Options ---
def version_callback(value: bool):
    if value:
        typer.echo(f"Compact Memory version: {__version__}")
        raise typer.Exit()


@app.callback(
    invoke_without_command=True
)  # invoke_without_command=True allows global options even if no subcommand is given
def main_callback(  # Renamed from main to main_callback for clarity
    ctx: typer.Context,
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Path to write debug logs. If not set, logs are not written to file.",
        resolve_path=True,  # Resolve path early
        show_default=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose (DEBUG level) logging to console and log file (if specified).",
    ),
    memory_path: Optional[
        str
    ] = typer.Option(  # Keep as str, resolve after Config loads
        None,
        "--memory-path",
        "-m",
        help="Path to Compact Memory engine storage. Overrides env var and config.",
        show_default=False,
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="Default model ID for LLM interactions. Overrides env var and config.",
        show_default=False,
    ),
    engine_id: Optional[
        str
    ] = typer.Option(  # This is for default history compression / one-shot compression
        None,
        "--engine",
        help="Default compression engine ID (e.g., for history, one-shot compress). Overrides env var/config.",
        show_default=False,
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Override LLM provider for this invocation (e.g., openai).",
        show_default=False,
    ),
    provider_url: Optional[str] = typer.Option(
        None,
        "--provider-url",
        help="Base URL for an OpenAI-compatible provider.",
        show_default=False,
    ),
    provider_key: Optional[str] = typer.Option(
        None,
        "--provider-key",
        help="API key for the OpenAI-compatible provider.",
        show_default=False,
    ),
    version: Optional[bool] = typer.Option(  # version_callback handles this
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,  # Eager means it runs before other processing
        help="Show the application version and exit.",
    ),
):
    """
    Compact Memory CLI main entry point.
    Manages global options like logging, memory path, and default model/engine IDs.
    """
    if ctx.obj is None:
        ctx.obj = {}

    # Initialize Config. This will load from files and environment variables.
    # CLI options provided to this main_callback will then override these.
    config = Config()  # Fresh config instance for this invocation

    # Configure logging based on CLI options or config defaults
    resolved_log_file = (
        log_file
        if log_file
        else Path(config.get("log_file")) if config.get("log_file") else None
    )
    resolved_verbose = (
        verbose if verbose is not None else config.get("verbose", False)
    )  # Default to False if not in config

    if resolved_log_file:
        # Ensure log_file path is absolute and parent dir exists
        resolved_log_file = resolved_log_file.resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        level = logging.DEBUG if resolved_verbose else logging.INFO
        configure_logging(
            str(resolved_log_file), level
        )  # configure_logging expects string path
    elif resolved_verbose:  # Verbose console logging even if no file
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:  # Default console logging if not verbose and no log file
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve paths and settings: CLI > Environment (via Config) > Config File (via Config) > Default (via Config)
    # 1. Memory Path
    if memory_path:  # CLI option for memory_path is directly provided
        resolved_memory_path_str = str(Path(memory_path).expanduser().resolve())
    else:  # Not provided via CLI, so rely on Config object (env, file, or default)
        resolved_memory_path_str = config.get("compact_memory_path")
        if resolved_memory_path_str:  # Ensure it's absolute if loaded from config/env
            resolved_memory_path_str = str(
                Path(resolved_memory_path_str).expanduser().resolve()
            )

    # 2. Default Model ID
    resolved_model_id = (
        model_id if model_id is not None else config.get("default_model_id")
    )

    # 3. Default Engine ID (for history compression / one-shot compress command)
    resolved_engine_id = (
        engine_id if engine_id is not None else config.get("default_engine_id")
    )

    resolved_provider = provider if provider is not None else None
    resolved_provider_url = provider_url if provider_url is not None else None
    resolved_provider_key = provider_key if provider_key is not None else None

    # Store resolved values in context for subcommands
    ctx.obj["config"] = config  # The Config instance itself
    ctx.obj["verbose"] = resolved_verbose
    ctx.obj["log_file"] = str(resolved_log_file) if resolved_log_file else None
    ctx.obj["compact_memory_path"] = resolved_memory_path_str
    ctx.obj["default_model_id"] = resolved_model_id
    ctx.obj["default_engine_id"] = resolved_engine_id
    ctx.obj["provider_override"] = resolved_provider
    ctx.obj["provider_url"] = resolved_provider_url
    ctx.obj["provider_key"] = resolved_provider_key

    # Load plugins after configuration is sorted out
    try:
        load_plugins()
        enable_all_experimental_engines()  # This handles contrib engines now
    except Exception as e:
        logging.error(
            f"Error during plugin loading or enabling experimental engines: {e}"
        )
        # Depending on severity, might raise typer.Exit(1)

    # Logic for prompting for memory_path if not set and command requires it
    # This needs to be evaluated carefully based on which commands truly need it upfront.
    # Most commands that need it can fetch it from ctx.obj and handle absence themselves.
    # Example: query, engine stats, engine init (target dir), engine clear, engine validate
    # Commands like config, dev list-*, compress (to stdout/file) might not need it.
    current_command_name = ctx.invoked_subcommand
    if current_command_name:
        # List of commands or command groups that might not need memory_path immediately
        # or handle its absence gracefully.
        commands_not_needing_memory_path_strict = [
            "config",
            "dev",
            "version",
        ]  # "version" is handled by callback

        # Check if the current command or its parent group is in the exclusion list
        requires_memory_path_check = True
        if any(
            current_command_name.startswith(cmd_prefix)
            for cmd_prefix in commands_not_needing_memory_path_strict
        ):
            requires_memory_path_check = False

        # Specific subcommands of 'dev' or 'engine' that don't need a path
        if current_command_name in [
            "dev.list-metrics",
            "dev.list-engines",
            "dev.list-strategies",
            "dev.create-engine-package",
            "dev.validate-engine-package",
            "dev.download-embedding-model",
            "dev.download-chat-model",
            "engine.list",
            "engine.info",  # These list available, don't operate on a store
        ]:
            requires_memory_path_check = False

        # Compress to stdout/file doesn't need a memory_path
        if (
            current_command_name == "compress" and not memory_path
        ):  # if compress is called without --memory-path global opt
            requires_memory_path_check = False

        if requires_memory_path_check and not resolved_memory_path_str:
            is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
            prompt_default_path_str = config.get_default(
                "compact_memory_path"
            )  # Get schema default

            if is_interactive:
                typer.secho(
                    "The Compact Memory path (engine store location) is not set.",
                    fg=typer.colors.YELLOW,
                )
                new_path_input = typer.prompt(
                    "Please enter the path for Compact Memory storage",
                    default=(
                        str(Path(prompt_default_path_str).expanduser())
                        if prompt_default_path_str
                        else None
                    ),
                )
                if new_path_input:
                    resolved_memory_path_str = str(
                        Path(new_path_input).expanduser().resolve()
                    )
                    ctx.obj["compact_memory_path"] = (
                        resolved_memory_path_str  # Update context
                    )
                    typer.secho(
                        f"Using memory path: {resolved_memory_path_str}",
                        fg=typer.colors.GREEN,
                    )
                    typer.echo(
                        f'To set this path permanently, run: compact-memory config set compact_memory_path "{resolved_memory_path_str}"'
                    )
                else:
                    typer.secho(
                        "Memory path is required to proceed with this command.",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=1)
            else:  # Non-interactive, path required but not set
                typer.secho(
                    "Error: Compact Memory path is not set and required for this command.",
                    fg=typer.colors.RED,
                    err=True,
                )
                typer.secho(
                    "Set via --memory-path, COMPACT_MEMORY_PATH env var, or 'compact-memory config set compact_memory_path ...'.",
                    err=True,
                )
                raise typer.Exit(code=1)


# Entry point for script execution (though Typer CLI handles this)
if __name__ == "__main__":
    app()
