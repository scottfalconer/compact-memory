import json
import sys
import time
import logging
import inspect  # Added
from pathlib import Path
from typing import Optional, Any
from dataclasses import asdict  # For trace output

import typer
from typer import Context  # Explicit import for type hint clarity

from compact_memory.config import Config  # Added
from compact_memory.llm_providers import get_llm_provider  # Added
from compact_memory.token_utils import token_count, SimpleTokenizer
from compact_memory.engines.registry import (
    get_compression_engine,
    available_engines,
    get_engine_metadata,
)
from compact_memory.engines import (
    BaseCompressionEngine,
    load_engine,
)  # For loading main engine
from compact_memory.engines import (
    CompressedMemory,
    CompressionTrace,
)  # Dataclasses for return types
from compact_memory.engines.pipeline_engine import (
    PipelineConfig,
    PipelineEngine,
)
from compact_memory.engine_config import EngineConfig
from compact_memory.exceptions import (
    ConfigurationError,
    EngineError,
    EngineLoadError,
    EngineSaveError,
    CompactMemoryError,
)

# PrototypeEngine was removed


# compress_app = typer.Typer() # Similar to query, can be a single command function for now


def compress_command(  # Renamed from compress
    ctx: typer.Context,
    text: Optional[str] = typer.Option(
        None,
        "--text",
        help="Raw text to compress or '-' to read from stdin",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        exists=True,  # Typer will check existence
        dir_okay=False,
        resolve_path=True,  # Resolve to absolute path
        help="Path to a single text file",
    ),
    dir_path: Optional[
        Path
    ] = typer.Option(  # Renamed from dir to dir_path to avoid conflict
        None,
        "--dir",  # Keep CLI argument as --dir
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Path to a directory of input files",
    ),
    *,  # Marks subsequent arguments as keyword-only
    engine_arg: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Specify the compression engine. Overrides the global default.",
    ),
    pipeline_config_str: Optional[str] = typer.Option(  # Added
        None,
        "--pipeline-config",
        help="JSON string defining the pipeline configuration when --engine is 'pipeline'.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        resolve_path=True,
        help="File path to write compressed output. If unspecified, prints to console. Cannot be used with --dir.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        resolve_path=True,
        help="Directory to write compressed output. When --dir is used, this specifies the directory for the combined 'compressed_output.txt'.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output compressed result in JSON format to stdout (not valid with --memory-path or --dir)",
    ),
    memory_path_arg: Optional[
        str
    ] = typer.Option(  # Stays string for flexibility, resolved to Path later
        None,
        "--memory-path",
        "-m",
        help="Path of the engine store directory to store the compressed content (instead of file/stdout).",
    ),
    output_trace: Optional[Path] = typer.Option(
        None,
        "--output-trace",
        resolve_path=True,
        help="File path to write the CompressionTrace JSON object. (Not applicable for directory input or --memory-path).",
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
        ...,  # Make budget mandatory
        "--budget",  # Added explicit option name
        "-b",  # Added short option name
        help="Token budget for the compressed output. The engine will aim to keep the output within this limit.",
    ),
    verbose_stats: bool = typer.Option(
        False,
        "--verbose-stats",
        help="Show detailed token counts and processing time per item.",
    ),
) -> None:
    """Compress text or files using a one-shot engine."""
    # Validation for pipeline_config_str (using engine_arg before it's finalized for default)
    if engine_arg == PipelineEngine.id and not pipeline_config_str:
        typer.secho(
            "Error: --pipeline-config is required when --engine is 'pipeline'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if engine_arg != PipelineEngine.id and pipeline_config_str:
        typer.secho(
            "Error: --pipeline-config can only be used when --engine is 'pipeline'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    final_engine_id = (
        engine_arg if engine_arg is not None else ctx.obj.get("default_engine_id")
    )
    if not final_engine_id:
        typer.secho(
            "Error: Compression engine not specified. Use --engine or set global default.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Attempt to get a tokenizer
    try:
        import tiktoken

        tokenizer_obj = tiktoken.get_encoding("gpt2")
    except ImportError:
        typer.secho(
            "Warning: tiktoken not found. Falling back to basic whitespace tokenizer.",
            fg=typer.colors.YELLOW,
        )
        tokenizer_obj = SimpleTokenizer()
    except Exception as e:
        typer.secho(
            f"Warning: Error initializing tiktoken: {e}. Falling back to basic whitespace tokenizer.",
            fg=typer.colors.YELLOW,
        )
        tokenizer_obj = SimpleTokenizer()

    # Validate input source
    input_sources = sum(x is not None for x in (text, file, dir_path))
    if input_sources != 1:
        typer.secho(
            "Error: Specify exactly ONE of --text, --file, or --dir.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Validate output options based on input source and memory_path_arg
    if memory_path_arg:
        if output_file or output_dir or json_output or output_trace:
            typer.secho(
                "Error: --memory-path cannot be combined with --output, --output-dir, --json, or --output-trace.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    elif dir_path:  # --dir is used, not --memory-path
        # Implement Option A: --output-file cannot be used with --dir
        if output_file:
            typer.secho(
                "Error: --output (--output-file) cannot be used with --dir. Use --output-dir to specify a directory for 'compressed_output.txt' or omit for default location.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        if json_output:
            typer.secho(
                "Error: --json output is not supported with --dir. Output is a single compressed file.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        # When outputting a directory we now allow a single trace file describing the combined compression.
        # output_trace will be passed through to the helper that writes results.
    else:  # Single text or file input, not --memory-path, not --dir
        if output_dir:  # This validation remains correct
            typer.secho(
                "Error: --output-dir is only valid with --dir.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    # Specific validations for --dir
    if dir_path is None:  # Not using --dir
        if recursive:
            typer.secho(
                "Error: --recursive is only valid with --dir.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        if pattern != "*.txt" and not dir_path:  # Pattern is only for --dir
            typer.secho(
                "Error: --pattern is only valid with --dir.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    else:  # Using --dir
        if "**" in pattern and not recursive:
            typer.secho(
                "Warning: Pattern includes '**' but --recursive not set; matching may miss subdirectories.",
                fg=typer.colors.YELLOW,
            )
        if not output_dir:
            typer.secho(
                "Info: No --output-dir specified for --dir input. 'compressed_output.txt' will be saved in the input directory.",
                fg=typer.colors.BLUE,
            )

    # --- Execution Logic ---
    if memory_path_arg:
        resolved_memory_path = Path(memory_path_arg).expanduser().resolve()
        try:
            # config_obj is ctx.obj["config"], default_model_id is ctx.obj["default_model_id"]
            # These are now passed down through ctx.
            main_engine_instance = load_engine(resolved_memory_path)
        except EngineLoadError as e:
            typer.secho(
                f"Error loading engine store from '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except CompactMemoryError as e:  # Other library errors during load
            typer.secho(
                f"Error related to engine store at '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except Exception as e:  # General fallback for load_engine
            typer.secho(
                f"An unexpected error occurred loading engine from '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        if text is not None:
            actual_text = sys.stdin.read() if text == "-" else text
            _compress_text_to_memory(
                main_engine_instance,
                actual_text,
                final_engine_id,
                budget,
                verbose_stats,
                tokenizer_obj,
                ctx,  # Pass context
                pipeline_config_str=pipeline_config_str,  # Added
            )
        elif file is not None:
            _compress_file_to_memory(
                main_engine_instance,
                file,
                final_engine_id,
                budget,
                verbose_stats,
                tokenizer_obj,
                ctx,  # Pass context
                pipeline_config_str=pipeline_config_str,  # Added
            )
        elif dir_path is not None:  # This was dir in the original, renamed to dir_path
            _compress_directory_to_memory(
                main_engine_instance,
                dir_path,
                final_engine_id,
                budget,
                recursive,
                pattern,
                verbose_stats,
                tokenizer_obj,
                ctx,  # Pass context
                pipeline_config_str=pipeline_config_str,  # Added
            )

        # Persist changes to the engine store
        try:
            main_engine_instance.save(resolved_memory_path)
            typer.secho(
                f"Content compressed and ingested. Engine store at '{resolved_memory_path}' saved.",
                fg=typer.colors.GREEN,
            )
        except EngineSaveError as e:
            typer.secho(
                f"Error saving engine store at '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except CompactMemoryError as e:  # Other library errors during save
            typer.secho(
                f"Error related to saving engine store at '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except Exception as e:  # General fallback for save
            typer.secho(
                f"An unexpected error occurred saving engine store at '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    else:  # Output to file or stdout
        if text is not None:
            actual_text = sys.stdin.read() if text == "-" else text
            _compress_text_to_file_or_stdout(
                actual_text,
                final_engine_id,
                budget,
                output_file,
                output_trace,
                verbose_stats,
                tokenizer_obj,
                json_output,
                ctx,  # Pass context
                pipeline_config_str=pipeline_config_str,  # Added
            )
        elif file is not None:
            _compress_file_to_file_or_stdout(
                file,
                final_engine_id,
                budget,
                output_file,
                output_trace,
                verbose_stats,
                tokenizer_obj,
                json_output,
                ctx,  # Pass context
                pipeline_config_str=pipeline_config_str,  # Added
            )
        elif dir_path is not None:  # This was dir in the original
            _compress_directory_to_files(
                dir_path,
                final_engine_id,
                budget,
                output_dir,
                output_trace,
                recursive,
                pattern,
                verbose_stats,
                tokenizer_obj,
                ctx,  # Pass context
                pipeline_config_str=pipeline_config_str,  # Added
            )


# --- Helper Functions (adapted from original CLI, now internal to this module) ---


def _get_one_shot_compression_engine(
    engine_id: str, ctx: typer.Context, pipeline_config_str: Optional[str] = None
) -> BaseCompressionEngine:
    """Helper to get and instantiate a one-shot compression engine."""
    if engine_id == PipelineEngine.id:
        if not pipeline_config_str:
            # This case should ideally be caught by the validation in compress_command,
            # but it's good to have a safeguard here.
            typer.secho(
                "Error: --pipeline-config is required when using the pipeline engine.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            pipeline_config_json = json.loads(pipeline_config_str)  # Parsed JSON object

            if not isinstance(pipeline_config_json, dict):
                typer.secho(
                    "Error: Pipeline config JSON must be a dictionary (object) with an 'engines' key.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)

            if "engines" not in pipeline_config_json:
                typer.secho(
                    "Error: Pipeline config JSON dictionary must contain an 'engines' key.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)

            engine_configs_list_json = pipeline_config_json["engines"]
            if not isinstance(engine_configs_list_json, list):
                typer.secho(
                    "Error: The 'engines' key in pipeline config JSON must contain a list of engine configurations.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)

            engine_configs = []
            for config_dict in engine_configs_list_json:
                if not isinstance(config_dict, dict):
                    typer.secho(
                        f"Error: Each item in the 'engines' list must be a dictionary, found: {type(config_dict)}",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=1)
                engine_configs.append(EngineConfig(**config_dict))

            pipeline_config_obj = PipelineConfig(engines=engine_configs)
            return PipelineEngine(
                pipeline_config_obj
            )  # Pass PipelineConfig to PipelineEngine
        except json.JSONDecodeError as e:
            typer.secho(
                f"Error decoding pipeline config JSON: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except TypeError as e:
            typer.secho(
                f"Error in pipeline engine configuration structure (e.g., invalid parameters for EngineConfig): {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(
                f"Error creating pipeline engine from config: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    try:
        EngineCls = get_compression_engine(engine_id)
        info = get_engine_metadata(engine_id)
        if info and info.get("source") == "contrib":  # 'contrib' source check
            typer.secho(
                f"\u26a0\ufe0f Using experimental one-shot compression engine '{engine_id}' from contrib.",
                fg=typer.colors.YELLOW,
            )

        # Inspect constructor for 'llm_provider'
        sig = inspect.signature(EngineCls.__init__)
        if "llm_provider" in sig.parameters:
            config_obj: Config = ctx.obj["config"]
            effective_model_id = ctx.obj.get("default_model_id")
            provider_override = ctx.obj.get("provider_override")
            provider_url = ctx.obj.get("provider_url")
            provider_key = ctx.obj.get("provider_key")

            if effective_model_id:
                provider = get_llm_provider(
                    effective_model_id,
                    config_obj,
                    provider_override=provider_override,
                    base_url=provider_url,
                    api_key=provider_key,
                )
                if provider:
                    return EngineCls(llm_provider=provider)
                else:
                    typer.secho(
                        f"Warning: Could not create LLM provider for model '{effective_model_id}'. Engine '{engine_id}' may not function as expected.",
                        fg=typer.colors.YELLOW,
                    )
                    return (
                        EngineCls()
                    )  # Fallback if provider creation fails but engine might work without
            else:  # No model_id specified, but engine expects a provider
                typer.secho(
                    f"Warning: Engine '{engine_id}' expects an LLM provider, but no model ID was specified (via --model-id or default).",
                    fg=typer.colors.YELLOW,
                )
                return EngineCls()  # Fallback
        else:  # Engine does not take llm_provider
            return EngineCls()
    except KeyError:
        typer.secho(
            f"Error: Unknown one-shot compression engine '{engine_id}'. Available: {', '.join(available_engines())}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


def _compress_text_core(
    text_content: str,
    engine_id: str,
    budget: int,
    tokenizer: Any,
    ctx: typer.Context,
    pipeline_config_str: Optional[str] = None,  # Added
) -> tuple[CompressedMemory, Optional[CompressionTrace], float]:
    """Core logic to compress text, returns compressed memory, trace, and time."""
    try:
        engine = _get_one_shot_compression_engine(engine_id, ctx, pipeline_config_str)
        start_time = time.time()
        compressed_mem: CompressedMemory = engine.compress(
            text_content, budget, tokenizer=tokenizer
        )
        elapsed_ms = (time.time() - start_time) * 1000

        if not isinstance(compressed_mem, CompressedMemory):
            # This case should ideally be caught by type checkers if engines adhere to the interface.
            # However, as a runtime check for safety:
            logging.error(
                f"Compression engine '{engine_id}' returned an unexpected result type: {type(compressed_mem)}"
            )
            typer.secho(
                f"Internal Error: Compression engine '{engine_id}' returned an unexpected result type.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    except ConfigurationError as e:
        typer.secho(
            f"Configuration error for engine '{engine_id}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    except EngineError as e:  # Catch other engine-specific errors during compress
        typer.secho(
            f"Error during compression with engine '{engine_id}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    except CompactMemoryError as e:  # Broader library errors
        typer.secho(
            f"An error occurred during compression: {e}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    except Exception as e:  # General fallback
        typer.secho(
            f"An unexpected error occurred during compression with engine '{engine_id}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        logging.exception(f"Compression failed with engine {engine_id}")
        raise typer.Exit(code=1)

    # Extract trace_obj from compressed_mem (this part remains the same)
    trace_obj: Optional[CompressionTrace] = compressed_mem.trace

    if not isinstance(compressed_mem, CompressedMemory):
        typer.secho(
            f"Error: Compression engine '{engine_id}' returned an unexpected result type: {type(compressed_mem)}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Extract trace_obj from compressed_mem
    trace_obj: Optional[CompressionTrace] = compressed_mem.trace

    # Populate processing_ms in the trace if the engine didn't set it
    if trace_obj and trace_obj.processing_ms is None:
        trace_obj.processing_ms = elapsed_ms
    elif compressed_mem.trace is None and elapsed_ms is not None:
        # If there was no trace object at all, but we have processing time,
        # we could potentially create a minimal trace here, but the current structure
        # expects engines to create their own traces. For now, if no trace, then no processing_ms is stored there.
        # However, elapsed_ms is still available for direct output if verbose_stats is on.
        pass

    return compressed_mem, trace_obj, elapsed_ms


def _output_results(
    original_text: str,
    compressed_memory: CompressedMemory,
    trace_obj: Optional[CompressionTrace],
    elapsed_ms: float,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    json_output: bool,
    source_name: str = "text",  # e.g. "text", "file: <filename>"
):
    """Handles outputting compressed text, trace, and stats."""
    if verbose_stats:
        orig_tokens = token_count(tokenizer, original_text)
        comp_tokens = token_count(tokenizer, compressed_memory.text)
        typer.echo(f"Source: {source_name}")
        typer.echo(
            f"Original tokens: {orig_tokens}\nCompressed tokens: {comp_tokens}\nTime ms: {elapsed_ms:.1f}"
        )

    if output_file:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(compressed_memory.text)
            typer.echo(f"Saved compressed output to {output_file}")
        except Exception as e:
            typer.secho(
                f"Error writing output to {output_file}: {e}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
    elif (
        json_output
    ):  # json_output implies not writing to a file (handled by arg validation)
        data = {
            "source": source_name,
            "compressed_text": compressed_memory.text,
            "original_tokens": token_count(tokenizer, original_text),
            "compressed_tokens": token_count(tokenizer, compressed_memory.text),
            "processing_ms": elapsed_ms,
            "trace": asdict(trace_obj) if trace_obj else None,
        }
        try:
            typer.echo(json.dumps(data, indent=2))
        except TypeError as e:  # Handle non-serializable trace details
            typer.secho(
                f"Error serializing JSON output (possibly from trace): {e}",
                err=True,
                fg=typer.colors.RED,
            )
            # Fallback: try to serialize without trace if it's the issue
            if trace_obj:
                data["trace"] = "Error: Trace not serializable"
            typer.echo(json.dumps(data, indent=2))

    else:  # Print to stdout
        typer.echo(compressed_memory.text)

    if trace_file and trace_obj:
        try:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(json.dumps(asdict(trace_obj), indent=2))
            typer.echo(f"Saved compression trace to {trace_file}")
        except Exception as e:
            typer.secho(
                f"Error writing trace to {trace_file}: {e}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)


def _compress_text_to_file_or_stdout(
    text_content: str,
    engine_id: str,
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    json_output: bool,
    ctx: typer.Context,  # Added context
    pipeline_config_str: Optional[str] = None,  # Added
) -> None:
    compressed_mem, trace_obj, elapsed_ms = _compress_text_core(
        text_content,
        engine_id,
        budget,
        tokenizer,
        ctx,
        pipeline_config_str,  # Pass pipeline_config_str
    )
    _output_results(
        "stdin/text arg",
        compressed_mem,
        trace_obj,
        elapsed_ms,
        output_file,
        trace_file,
        verbose_stats,
        tokenizer,
        json_output,
    )


def _compress_file_to_file_or_stdout(
    file_path: Path,
    engine_id: str,
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    json_output: bool,
    ctx: typer.Context,  # Added context
    pipeline_config_str: Optional[str] = None,  # Added
) -> None:
    try:
        text_content = file_path.read_text()
    except Exception as e:
        typer.secho(
            f"Error reading file {file_path}: {e}", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(1)

    # If output_file is not specified, and not json_output, derive it for single file processing
    # This behavior is slightly different from original for clarity: output is explicit or stdout.
    # Original might have created suffixed file by default.
    # For this refactor, if no -o, it goes to stdout unless json_output.

    compressed_mem, trace_obj, elapsed_ms = _compress_text_core(
        text_content,
        engine_id,
        budget,
        tokenizer,
        ctx,
        pipeline_config_str,  # Pass pipeline_config_str
    )
    _output_results(
        f"file: {file_path.name}",
        compressed_mem,
        trace_obj,
        elapsed_ms,
        output_file,
        trace_file,
        verbose_stats,
        tokenizer,
        json_output,
    )


def _compress_directory_to_files(
    dir_path_obj: Path,
    engine_id: str,
    budget: int,  # Renamed dir_path to dir_path_obj
    output_dir_obj: Optional[Path],
    trace_file_obj: Optional[Path],
    recursive: bool,
    pattern: str,  # Renamed output_dir to output_dir_obj
    verbose_stats: bool,
    tokenizer: Any,
    ctx: typer.Context,  # Added context
    pipeline_config_str: Optional[str] = None,  # Added
) -> None:
    files_to_process = sorted(
        list(dir_path_obj.rglob(pattern) if recursive else dir_path_obj.glob(pattern))
    )
    if not files_to_process:
        typer.echo(f"No files matching pattern '{pattern}' found in '{dir_path_obj}'.")
        return

    combined_content = []
    total_files_read = 0
    total_files_skipped = 0

    typer.echo(
        f"Reading and combining files from '{dir_path_obj}' matching '{pattern}'..."
    )
    for input_file_path in files_to_process:
        if not input_file_path.is_file():
            continue  # Skip subdirectories

        try:
            # typer.echo(f"  Reading {input_file_path}...") # Optional: too verbose if many files
            text_content = input_file_path.read_text()
            combined_content.append(text_content)
            total_files_read += 1
        except Exception as e:
            typer.secho(
                f"  Warning: Error reading file {input_file_path}, skipping: {e}",
                fg=typer.colors.YELLOW,
            )
            total_files_skipped += 1
            continue

    if not combined_content:
        typer.secho(
            f"No content could be read from files in '{dir_path_obj}' matching '{pattern}'.",
            fg=typer.colors.RED,
        )
        return

    full_text_content = "\n\n".join(
        combined_content
    )  # Join with double newline to separate file contents

    typer.echo(
        f"Successfully read and combined content from {total_files_read} file(s). Skipped {total_files_skipped} file(s) due to errors."
    )
    typer.echo(f"Compressing combined content...")

    compressed_mem, trace_obj, elapsed_ms = _compress_text_core(
        full_text_content,
        engine_id,
        budget,
        tokenizer,
        ctx,
        pipeline_config_str,  # Pass pipeline_config_str
    )

    # Determine output path
    output_file_name = "compressed_output.txt"
    if output_dir_obj:
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        final_output_path = output_dir_obj / output_file_name
    else:
        final_output_path = dir_path_obj / output_file_name

    source_name = (
        f"directory: {dir_path_obj} (pattern: {pattern}, {total_files_read} files)"
    )

    # Output the single compressed file
    # trace_file is None for dir mode as per original logic and requirements.
    # json_output is False for dir mode as per original logic and requirements.
    _output_results(
        full_text_content,  # Original text is the combined content
        compressed_mem,
        trace_obj,  # This will be None if the engine doesn't return it, or the actual trace
        elapsed_ms,
        final_output_path,
        trace_file_obj,
        verbose_stats,
        tokenizer,
        False,  # json_output is False for directory mode
        source_name=source_name,
    )

    if verbose_stats or total_files_read > 0:
        typer.echo(
            f"Finished processing directory. Combined {total_files_read} file(s) into '{final_output_path}'."
        )


# --- Helpers for --memory-path ---


def _compress_text_to_memory(
    main_engine_instance: BaseCompressionEngine,
    text_to_compress: str,  # Renamed variables
    one_shot_engine_id: str,
    budget: int,  # Renamed variables
    verbose_stats: bool,
    tokenizer: Any,
    ctx: typer.Context,  # Added context
    pipeline_config_str: Optional[str] = None,  # Added
    source_document_id: Optional[str] = "text_input",  # Default ID
) -> None:
    compressed_mem, _, elapsed_ms = _compress_text_core(
        text_to_compress,
        one_shot_engine_id,
        budget,
        tokenizer,
        ctx,
        pipeline_config_str=pipeline_config_str,  # Pass pipeline_config_str
    )

    # Ingest into the main engine instance
    # This part depends on how main_engine_instance expects to ingest pre-compressed content.
    # Assuming a method like `add_compressed_memory` or adapting `ingest`.
    # The original code used `main_engine.add_memory` for PrototypeEngine or `main_engine.ingest` for others.
    # PrototypeEngine was removed, defaulting to standard ingest.
    # Generic ingest, assuming it can take a string.
    # If engines need more structured input for pre-compressed text, this needs adjustment.
    main_engine_instance.ingest(compressed_mem.text)
    # Or if there's a specific method for pre-compressed:
    # main_engine_instance.ingest_compressed(compressed_mem, source_id=source_document_id)

    if verbose_stats:
        orig_tokens = token_count(tokenizer, text_to_compress)
        comp_tokens = token_count(tokenizer, compressed_mem.text)
        typer.echo(f"Source: {source_document_id}")
        typer.echo(
            f"Original tokens: {orig_tokens}\nCompressed tokens (one-shot): {comp_tokens}\nTime (one-shot) ms: {elapsed_ms:.1f}"
        )
        typer.echo(
            f"Compressed text ingested into main engine at '{main_engine_instance.path if hasattr(main_engine_instance, 'path') else 'memory store'}'."
        )


def _compress_file_to_memory(
    main_engine_instance: BaseCompressionEngine,
    file_path_obj: Path,  # Renamed
    one_shot_engine_id: str,
    budget: int,
    verbose_stats: bool,
    tokenizer: Any,
    ctx: typer.Context,  # Added context
    pipeline_config_str: Optional[str] = None,  # Added
) -> None:
    try:
        text_content = file_path_obj.read_text()
    except Exception as e:
        typer.secho(
            f"Error reading file {file_path_obj} for memory compression: {e}",
            err=True,
            fg=typer.colors.YELLOW,
        )
        return

    _compress_text_to_memory(
        main_engine_instance,
        text_content,
        one_shot_engine_id,
        budget,
        verbose_stats,
        tokenizer,
        ctx,  # Pass context
        pipeline_config_str=pipeline_config_str,  # Pass pipeline_config_str
        source_document_id=str(file_path_obj.name),
    )


def _compress_directory_to_memory(
    main_engine_instance: BaseCompressionEngine,
    dir_path_obj: Path,  # Renamed
    one_shot_engine_id: str,
    budget: int,
    recursive: bool,
    pattern: str,
    verbose_stats: bool,
    tokenizer: Any,
    ctx: typer.Context,  # Added context
    pipeline_config_str: Optional[str] = None,  # Added
) -> None:
    files_to_process = list(
        dir_path_obj.rglob(pattern) if recursive else dir_path_obj.glob(pattern)
    )
    if not files_to_process:
        typer.echo(
            f"No files matching pattern '{pattern}' found in '{dir_path_obj}' for memory compression."
        )
        return

    processed_count = 0
    for input_file_path in files_to_process:
        if not input_file_path.is_file():
            continue
        typer.echo(f"Processing {input_file_path} for memory compression...")
        _compress_file_to_memory(
            main_engine_instance,
            input_file_path,
            one_shot_engine_id,
            budget,
            verbose_stats,
            tokenizer,
            ctx,  # Pass context
            pipeline_config_str=pipeline_config_str,  # Pass pipeline_config_str
        )
        processed_count += 1

    if verbose_stats or processed_count > 0:
        typer.echo(
            f"Finished directory processing for memory. Processed {processed_count} files."
        )


# Final check of imports and functions:
# - `dataclasses.asdict` is used.
# - `compact_memory.token_utils.token_count` is correctly pathed.
# - `compact_memory.engines.registry` for engine getters.
# - `compact_memory.engines.BaseCompressionEngine`, `load_engine`, `CompressedMemory`, `CompressionTrace` are pathed.
# - Standard library imports are fine.
# - Typer is fine.
# - All helper functions are prefixed with `_` to indicate they are internal.
# - Command function `compress_command` has renamed parameters to avoid conflict (e.g., `dir` -> `dir_path`).
# - Logic for handling different input/output combinations seems more robust.
# - Tokenizer fallback is included.
# - `budget` is now a required option.
# - `_compress_text_core` centralizes the actual compression call.
# - `_output_results` centralizes output logic.
# - Helpers for `--memory-path` are separated.
# - `source_document_id` is used more consistently.
# - Error messages and status updates improved.
# Looks reasonable.
