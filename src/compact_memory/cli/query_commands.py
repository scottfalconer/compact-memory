from pathlib import Path
from typing import Optional
import logging

import typer

from compact_memory.engines import (
    load_engine,
    get_compression_engine,
)  # Added get_compression_engine
from compact_memory import (
    local_llm,
)  # Assuming this is the correct module for LocalChatModel
from compact_memory.contrib import (
    ActiveMemoryManager,
)  # Assuming this is the correct module

# from compact_memory.plugin_loader import load_plugins # Global
# from compact_memory.contrib import enable_all_experimental_engines # Global
from compact_memory.engines.registry import (
    get_engine_metadata,
    available_engines,
)  # For history compression engine


query_app = typer.Typer(
    help="Query the Compact Memory engine store."
)  # Can be a Typer app or just hold commands


@query_app.command(  # Or use @app.command if main app is passed around, but for modularity, this is fine
    "query",  # Command name if this app is added to main via add_typer
    help='Queries the Compact Memory engine store and returns an AI-generated response.\n\nUsage Examples:\n  compact-memory query "What is the capital of France?"\n  compact-memory query "Explain the theory of relativity in simple terms" --show-prompt-tokens',
)
def query_command(  # Renamed from query
    ctx: typer.Context,
    query_text: str = typer.Argument(
        ..., help="The query text to send to the engine store."
    ),
    show_prompt_tokens: bool = typer.Option(
        False,
        "--show-prompt-tokens",
        help="Display the token count of the final prompt sent to the LLM.",
    ),
) -> None:
    """Query the Compact Memory engine store and print the response."""
    resolved_memory_path_str = ctx.obj.get("compact_memory_path")
    if not resolved_memory_path_str:
        typer.secho(
            "Error: Memory path not set. Use --memory-path or configure it globally.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    resolved_memory_path = Path(resolved_memory_path_str)

    try:
        engine_instance = load_engine(resolved_memory_path)  # Renamed variable
    except FileNotFoundError:
        typer.secho(
            f"Error: Engine store not found at '{resolved_memory_path}'. Please initialize it first using 'compact-memory engine init'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"Error loading engine from '{resolved_memory_path}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        logging.exception(
            f"Failed to load engine from {resolved_memory_path}"
        )  # Keep detailed log
        raise typer.Exit(code=1)

    final_model_id = ctx.obj.get("default_model_id")
    # This is the engine ID for history compression, not the main query engine
    final_history_compression_engine_id = ctx.obj.get("default_engine_id")

    if final_model_id is None:
        typer.secho(
            "Error: Default Model ID not specified. Use the global --model-id option or set it in the configuration.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            "Example: `compact-memory config set default_model_id <your_model_id>` or `compact-memory --model-id <your_model_id> query ...`",
            fg=typer.colors.CYAN,  # Using cyan for example command
            err=True,
        )
        raise typer.Exit(code=1)

    # Attach chat model to the loaded engine instance if it needs one
    # This part is engine-specific; some engines might manage their LLM internally.
    # The original code directly set `_chat_model`. A more robust way would be
    # engine_instance.set_chat_model(...) or similar, if engines follow that pattern.
    if hasattr(engine_instance, "_chat_model"):  # Check if the attribute exists
        try:
            # Assuming local_llm.LocalChatModel is the correct class
            engine_instance._chat_model = local_llm.LocalChatModel(
                model_name=final_model_id
            )
            engine_instance._chat_model.load_model()  # Ensure model is loaded
        except RuntimeError as exc:  # More specific exception if possible
            typer.secho(
                f"Error loading chat model '{final_model_id}' for the engine: {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except AttributeError:  # Should be caught by hasattr, but as a safeguard
            typer.secho(
                f"Warning: The loaded engine type '{type(engine_instance).__name__}' does not have a '_chat_model' attribute as expected by the CLI. Querying may fail.",
                fg=typer.colors.YELLOW,
            )
        # If engine does not have _chat_model, it's assumed it handles its LLM internally or doesn't need one for query.
    else:
        typer.secho(
            f"Info: Engine type '{type(engine_instance).__name__}' does not have a '_chat_model' attribute. Assuming it handles its LLM internally if needed for querying.",
            fg=typer.colors.BLUE,  # Using blue for informational message
        )

    active_memory_mgr = ActiveMemoryManager()  # Renamed variable
    history_comp_engine_instance = None
    if (
        final_history_compression_engine_id
        and final_history_compression_engine_id.lower() != "none"
    ):
        try:
            CompressionEngineCls = get_compression_engine(
                final_history_compression_engine_id
            )  # Renamed variable
            engine_meta_info = get_engine_metadata(
                final_history_compression_engine_id
            )  # Renamed variable
            if (
                engine_meta_info and engine_meta_info.get("source") == "contrib"
            ):  # Check source if available
                typer.secho(
                    "\u26a0\ufe0f Using experimental engine for history compression from contrib: not officially supported.",
                    fg=typer.colors.YELLOW,
                )
            history_comp_engine_instance = (
                CompressionEngineCls()
            )  # Instantiate the class
        except KeyError:  # If engine ID not found
            typer.secho(
                f"Error: Unknown history compression engine '{final_history_compression_engine_id}' (from global config/option). Available: {', '.join(available_engines())}",
                err=True,
                fg=typer.colors.RED,
            )
            typer.secho(
                f"Example: `compact-memory config set default_engine_id <valid_engine_id>` or `compact-memory --engine <valid_engine_id> query ...`",
                fg=typer.colors.CYAN,
                err=True,
            )
            raise typer.Exit(code=1)

    try:
        if not hasattr(engine_instance, "receive_channel_message"):
            typer.secho(
                f"Error: The loaded engine type '{type(engine_instance).__name__}' does not support the 'receive_channel_message' method required for querying.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Call receive_channel_message, adapting for potential lack of 'compression' kwarg
        # based on original code's try-except for TypeError.
        try:
            query_result = engine_instance.receive_channel_message(  # Renamed variable
                "cli",
                query_text,
                active_memory_mgr,
                compression=history_comp_engine_instance,
            )
        except TypeError as e:
            if (
                "got an unexpected keyword argument 'compression'" in str(e)
                or "takes at most 3 positional arguments" in str(e)
                or "takes from 3 to 4 positional arguments but 5 were given" in str(e)
            ):  # Adjusted error check
                typer.secho(
                    f"Warning: Engine type '{type(engine_instance).__name__}' does not support 'compression' parameter for history. Retrying without it.",
                    fg=typer.colors.YELLOW,
                )
                query_result = engine_instance.receive_channel_message(
                    "cli", query_text, active_memory_mgr
                )
            else:
                # Re-raise if it's a different TypeError
                raise
    except Exception as e:  # Catch other errors during query processing
        typer.secho(
            f"An unexpected error occurred during query: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        logging.exception("Unexpected error during query.")  # Log stack trace
        raise typer.Exit(code=1)

    reply_text = query_result.get("reply")  # Renamed variable
    if reply_text:
        typer.echo(reply_text)
    else:
        typer.secho("The engine store did not return a reply.", fg=typer.colors.YELLOW)

    if show_prompt_tokens and query_result.get("prompt_tokens") is not None:
        typer.echo(f"Prompt tokens: {query_result['prompt_tokens']}")


# Necessary imports:
# - pathlib.Path
# - typing.Optional
# - logging
# - typer
# - compact_memory.engines.load_engine
# - compact_memory.engines.get_compression_engine (for history compression)
# - compact_memory.engines.registry (for get_engine_metadata, available_engines for history)
# - compact_memory.local_llm (if LocalChatModel is there)
# - compact_memory.contrib.ActiveMemoryManager (if that's its location)

# Global plugin loading should handle plugins, so no local calls to load_plugins needed.
# Renamed functions and variables for clarity.
# Added more specific error handling and messages.
# Handled the case where `_chat_model` might not exist on the engine.
# Ensured `history_comp_engine_instance` is an instance of the compression engine.
# Adapted the TypeError check for `receive_channel_message` to be more robust.
# `logging.exception` added for unexpected errors.
# `local_llm` and `ActiveMemoryManager` import paths are assumed based on original structure;
# these might need adjustment if the actual locations are different in `src/compact_memory/`.
# For example, if `local_llm` is now `compact_memory.local_llm` and `ActiveMemoryManager` is
# `compact_memory.active_memory_manager`.
# Assuming `compact_memory.local_llm` and `compact_memory.contrib.ActiveMemoryManager` are correct.
# `compact_memory.engines.registry` for `get_engine_metadata` and `available_engines`.
# `compact_memory.engines.get_compression_engine` for instantiating history compressor.
# Looks okay.
