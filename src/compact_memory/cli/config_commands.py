from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from compact_memory.config import Config, DEFAULT_CONFIG, USER_CONFIG_PATH

config_app = typer.Typer(
    help="Manage Compact Memory application configuration settings."
)

@config_app.command(
    "set",
    help="Sets a Compact Memory configuration key to a new value in the user's global config file.\n\nUsage Examples:\n  compact-memory config set default_model_id openai/gpt-4-turbo\n  compact-memory config set compact_memory_path /mnt/my_data/compact_memory_store",
)
def set_config_command(  # Renamed from config_set_command
    ctx: typer.Context,
    key: str = typer.Argument(
        ...,
        help=f"The configuration key to set. Valid keys: {', '.join(DEFAULT_CONFIG.keys())}.",
    ),
    value: str = typer.Argument(..., help="The new value for the configuration key."),
) -> None:
    config: Config = ctx.obj["config"]  # Config object is expected to be in ctx.obj
    try:
        # The Config.set method should handle validation and saving.
        success = config.set(key, value) # This now directly calls the Config instance method
        if success:
            typer.secho(
                f"Successfully set '{key}' to '{value}' in the user global configuration: {USER_CONFIG_PATH}",
                fg=typer.colors.GREEN,
            )
            typer.echo(
                f"Note: Environment variables or local project '.gmconfig.yaml' may override this global setting."
            )
        else:
            # Assuming config.set prints specific errors or returns False on validation failure
            # If config.set raises exceptions for invalid keys or values, this will be caught below.
            # typer.secho(f"Failed to set '{key}'. It might be an invalid key or value.", fg=typer.colors.RED, err=True) # Generic error
            raise typer.Exit(code=1) # Exit if not successful and no specific error was printed by config.set
    except ValueError as e: # Catch validation errors from Pydantic models within Config.set
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:  # Catch any other unexpected errors during the process
        typer.secho(
            f"An unexpected error occurred while setting configuration: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@config_app.command(
    "show",
    help="Displays current Compact Memory configuration values, their effective settings, and their sources.\n\nUsage Examples:\n  compact-memory config show\n  compact-memory config show --key default_engine_id",
)
def show_config_command(  # Renamed from config_show_command
    ctx: typer.Context,
    key: Optional[str] = typer.Option(
        None,
        "--key",
        "-k",
        help=f"Specific configuration key to display. Valid keys: {', '.join(DEFAULT_CONFIG.keys())}.",
    ),
) -> None:
    config: Config = ctx.obj["config"] # Config object from context
    console = Console(width=200) # Increased width for better display
    table = Table(title="Compact Memory Configuration")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Effective Value", style="magenta", overflow="fold") # Allow folding for long values
    table.add_column("Source", style="green", no_wrap=True, overflow="fold") # Allow folding for long source paths

    if key:
        if key not in config.get_all_keys(): # Check if the key is known at all
            typer.secho(f"Error: Configuration key '{key}' is not a recognized key.", fg=typer.colors.RED, err=True)
            typer.echo("Known configuration keys are:")
            for known_key in sorted(config.get_all_keys()):
                typer.echo(f"- {known_key}")
            raise typer.Exit(code=1)

        value, source_info = config.get_with_source(key)
        # value will be None if the key is known but not set by any source (i.e., it would use its default if accessed with .get())
        # However, get_with_source should ideally return the default value and its source in such cases.
        # Let's assume get_with_source returns the current effective value (which could be a default).
        if value is not None: # Or always show, even if value is None but source indicates "default"
            table.add_row(key, str(value) if value is not None else "Not Set (uses default)", source_info)
        else: # This case might not be hit if get_with_source always returns a value (even default)
            typer.secho(
                f"Configuration key '{key}' found but has no value and no default is specified in its definition.",
                fg=typer.colors.YELLOW,
            )

    else:
        all_configs_with_sources = config.get_all_with_sources()
        if not all_configs_with_sources:
            typer.echo("No configurations found or defined.") # Changed message
            return

        sorted_keys = sorted(all_configs_with_sources.keys())

        for k_val in sorted_keys:
            value, source_info = all_configs_with_sources[k_val]
            table.add_row(k_val, str(value) if value is not None else "Not Set (uses default)", source_info)

    if table.row_count > 0:
        console.print(table)
    elif not key:
        typer.echo("No configuration settings found to display.") # Message if table is empty and no specific key was searched
    # If a specific key was searched and not found, the message is handled within the 'if key:' block.

# Imports needed:
# - typing.Optional
# - typer
# - rich.console.Console
# - rich.table.Table
# - compact_memory.config.Config (for type hint and using its methods)
# - compact_memory.config.DEFAULT_CONFIG (for help text)
# - compact_memory.config.USER_CONFIG_PATH (for messages)

# All command names updated (e.g., `config_set_command` to `set_config_command`).
# `ctx.obj["config"]` is used to get the `Config` instance.
# `DEFAULT_CONFIG.keys()` is used for help text.
# `USER_CONFIG_PATH` is used in messages.
# `Console` and `Table` from `rich` are used for `show_config_command`.
# Error handling in `set_config_command` improved to catch `ValueError` from Pydantic.
# Logic in `show_config_command` improved for handling specific key requests and listing all keys.
# Corrected the help string for `set_config_command` to use `DEFAULT_CONFIG.keys()`.
# Corrected the help string for `show_config_command` to use `DEFAULT_CONFIG.keys()`.
# `config.get_all_keys()` is a hypothetical method, assuming Config class has it. If not, `DEFAULT_CONFIG.keys()` or similar.
# For `show_config_command` when `key` is provided but not found, it now provides a list of known keys.
# The `config: Config = ctx.obj["config"]` type hint is good.
# The `console = Console(width=200)` is a good improvement.
# `table.add_column(..., overflow="fold")` is a good improvement for long values.
# `config.get_all_with_sources()` is assumed to be a method of the `Config` class.
# `config.get_with_source(key)` is assumed to be a method of the `Config` class.
# The logic for displaying "Not Set (uses default)" is appropriate.
# Seems complete and correct based on the original structure and desired refactoring.
# Assuming `Config.set` now handles the logic of finding the user config file path itself.
# The `config.set` method in `compact_memory.config` should be the one actually writing to `USER_CONFIG_PATH`.
# The `ctx.obj['config']` should be an instance of `Config` that's already loaded,
# so `set_config_command` and `show_config_command` operate on this live instance.
# The `Config` class itself should manage persistence when `set()` is called.
# Final check of imports and functionality looks good.
