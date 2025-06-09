from pathlib import Path
from typing import Optional
import logging

import typer

from compact_memory.engines import (
    load_engine,
    get_compression_engine,
)
from compact_memory import local_llm
from compact_memory.exceptions import CompactMemoryError, EngineLoadError # Import custom exceptions

from compact_memory.engines.registry import (
    get_engine_metadata,
    available_engines,
)


query_app = typer.Typer(help="Query the Compact Memory engine store.")


@query_app.command(
    "query",
    help='Queries the Compact Memory engine store and returns an AI-generated response.\n\nUsage Examples:\n  compact-memory query "What is the capital of France?"\n  compact-memory query "Explain the theory of relativity in simple terms" --show-prompt-tokens',
)
def query_command(
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
        engine_instance = load_engine(resolved_memory_path)
    except EngineLoadError as e:
        typer.secho(f"Error loading engine from '{resolved_memory_path}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except CompactMemoryError as e: # Catch other specific library errors
        typer.secho(f"A Compact Memory error occurred while loading the engine from '{resolved_memory_path}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e: # General fallback for unexpected errors
        typer.secho(
            f"An unexpected error occurred loading engine from '{resolved_memory_path}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        logging.exception(f"Failed to load engine from {resolved_memory_path}")
        raise typer.Exit(code=1)

    final_model_id = ctx.obj.get("default_model_id")
    final_history_compression_engine_id = ctx.obj.get("default_engine_id")

    if final_model_id is None:
        typer.secho(
            "Error: Default Model ID not specified. Use the global --model-id option or set it in the configuration.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            "Example: `compact-memory config set default_model_id <your_model_id>` or `compact-memory --model-id <your_model_id> query ...`",
            fg=typer.colors.CYAN,
            err=True,
        )
        raise typer.Exit(code=1)

    if hasattr(engine_instance, "_chat_model"):
        try:
            engine_instance._chat_model = local_llm.LocalChatModel(
                model_name=final_model_id
            )
            engine_instance._chat_model.load_model()
        except RuntimeError as exc:
            typer.secho(
                f"Error loading chat model '{final_model_id}' for the engine: {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except AttributeError:
            typer.secho(
                f"Warning: The loaded engine type '{type(engine_instance).__name__}' does not have a '_chat_model' attribute as expected by the CLI. Querying may fail.",
                fg=typer.colors.YELLOW,
            )
    else:
        typer.secho(
            f"Info: Engine type '{type(engine_instance).__name__}' does not have a '_chat_model' attribute. Assuming it handles its LLM internally if needed for querying.",
            fg=typer.colors.BLUE,
        )

    history_comp_engine_instance = None
    if (
        final_history_compression_engine_id
        and final_history_compression_engine_id.lower() != "none"
    ):
        try:
            CompressionEngineCls = get_compression_engine(
                final_history_compression_engine_id
            )
            engine_meta_info = get_engine_metadata(final_history_compression_engine_id)
            if engine_meta_info and engine_meta_info.get("source") == "contrib":
                typer.secho(
                    "\u26a0\ufe0f Using experimental engine for history compression from contrib: not officially supported.",
                    fg=typer.colors.YELLOW,
                )
            history_comp_engine_instance = CompressionEngineCls()
        except KeyError:
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
        try:
            query_result = engine_instance.receive_channel_message(
                "cli",
                query_text,
                compression=history_comp_engine_instance,
            )
        except TypeError as e:
            if (
                "got an unexpected keyword argument 'compression'" in str(e)
                or "takes at most 2 positional arguments" in str(e)
                or "takes from 2 to 3 positional arguments but 4 were given" in str(e)
            ):
                typer.secho(
                    f"Warning: Engine type '{type(engine_instance).__name__}' does not support 'compression' parameter for history. Retrying without it.",
                    fg=typer.colors.YELLOW,
                )
                query_result = engine_instance.receive_channel_message(
                    "cli", query_text
                )
            else:
                raise
    except CompactMemoryError as e: # Catch library-specific errors during query processing
        typer.secho(f"An error occurred during query processing: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e: # General fallback
        typer.secho(
            f"An unexpected error occurred during query: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        logging.exception("Unexpected error during query.")
        raise typer.Exit(code=1)

    reply_text = query_result.get("reply")
    if reply_text:
        typer.echo(reply_text)
    else:
        typer.secho("The engine store did not return a reply.", fg=typer.colors.YELLOW)

    if show_prompt_tokens and query_result.get("prompt_tokens") is not None:
        typer.echo(f"Prompt tokens: {query_result['prompt_tokens']}")


# Notes from original file structure for context:
# - compact_memory.engines.registry for get_engine_metadata, available_engines.
# - compact_memory.engines.get_compression_engine for history compressor.
# These comments are just for context; the code should be functional.
