import json
import sys
import time
from pathlib import Path
from typing import Optional, Any, List, Dict
from dataclasses import asdict  # For trace output

import typer

from compact_memory.config import Config
from compact_memory.llm_providers_abc import LLMProvider
from compact_memory.llm_providers.factory import create_llm_provider
from compact_memory.token_utils import token_count  # Assuming this is its new location
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
from compact_memory.prototype_engine import (
    PrototypeEngine,
)  # For type checking if needed for specific ingest


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
    # Output Options
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
    app_config: Config = ctx.obj.get("config")
    if not app_config:
        app_config = Config()
        typer.secho("Warning: Global app config not found in context, loaded a new one.", fg=typer.colors.YELLOW)

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

    # --- LLM Setup and Engine Instantiation ---
    llm_provider_instance: Optional[LLMProvider] = None
    engine_override_config: Dict[str, Any] = {}

    try:
        EngineCls = get_compression_engine(final_engine_id)
    except KeyError:
        typer.secho(
            f"Error: Unknown one-shot compression engine '{final_engine_id}'. Available: {', '.join(available_engines())}",
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
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        if llm_provider_type and not llm_model_name:
            typer.secho(
                "Error: --llm-model-name must be provided if --llm-provider-type is specified.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        if llm_model_name and not llm_provider_type and not llm_config:
            typer.secho(
                "Error: --llm-provider-type must be provided if --llm-model-name is specified without --llm-config.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        actual_llm_config_name = llm_config
        actual_llm_provider_type = llm_provider_type
        actual_llm_model_name = llm_model_name
        actual_llm_api_key = llm_api_key

        if not actual_llm_config_name and not actual_llm_provider_type:
            default_model_id = app_config.get("default_model_id")
            if default_model_id:
                typer.secho(f"No LLM specified directly; using default from config: {default_model_id}", fg=typer.colors.BLUE)
                if default_model_id in app_config.get_all_llm_configs():
                    actual_llm_config_name = default_model_id
                elif '/' in default_model_id:
                    parts = default_model_id.split('/', 1)
                    actual_llm_provider_type = parts[0]
                    actual_llm_model_name = parts[1]
                else:
                    typer.secho(
                        f"Error: Default model ID '{default_model_id}' is not a valid named configuration "
                        "or 'provider/model_name' format.",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=1)
            else:
                typer.secho(
                    f"Error: Engine '{final_engine_id}' requires an LLM. Please provide LLM configuration "
                    "via --llm-config or --llm-provider-type/--llm-model-name, or set a default_model_id "
                    "in the global config.",
                    fg=typer.colors.RED,
                    err=True,
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
            typer.secho(f"Error creating LLM provider: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        if final_engine_id == "readagent_gist":
            if readagent_gist_model_name is not None:
                engine_override_config["gist_model_name"] = readagent_gist_model_name
            elif actual_llm_model_name:
                 engine_override_config["gist_model_name"] = actual_llm_model_name

            if readagent_gist_length is not None:
                engine_override_config["gist_length"] = readagent_gist_length

            if readagent_lookup_max_tokens is not None:
                engine_override_config["lookup_max_tokens"] = readagent_lookup_max_tokens

            if readagent_qa_model_name is not None:
                engine_override_config["qa_model_name"] = readagent_qa_model_name
            elif actual_llm_model_name:
                engine_override_config["qa_model_name"] = actual_llm_model_name

            if readagent_qa_max_new_tokens is not None:
                engine_override_config["qa_max_new_tokens"] = readagent_qa_max_new_tokens

    engine_info = get_engine_metadata(final_engine_id)
    if engine_info and engine_info.get("source") == "contrib":
        typer.secho(
            f"\u26a0\ufe0f Using experimental one-shot compression engine '{final_engine_id}' from contrib.",
            fg=typer.colors.YELLOW,
        )

    engine_instance: BaseCompressionEngine
    if engine_requires_llm:
        if not llm_provider_instance:
            typer.secho("Critical error: LLM provider not initialized when required.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        engine_instance = EngineCls(config=engine_override_config, llm_provider=llm_provider_instance)
    else:
        engine_instance = EngineCls()

    # --- Tokenizer Setup ---
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")  # Default to gpt2 encoding
        tokenizer_func = lambda text, **k: {
            "input_ids": enc.encode(text)
        }  # Adapt to expected structure
    except ImportError:
        typer.secho(
            "Warning: tiktoken not found. Falling back to basic whitespace tokenization for stats.",
            fg=typer.colors.YELLOW,
        )
        tokenizer_func = lambda text, **k: {"input_ids": text.split()}  # Basic fallback
    except Exception as e:
        typer.secho(
            f"Warning: Error initializing tiktoken: {e}. Falling back to basic whitespace tokenization.",
            fg=typer.colors.YELLOW,
        )
        tokenizer_func = lambda text, **k: {"input_ids": text.split()}

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
        if (
            output_file and not output_dir
        ):  # individual output file doesn't make sense for dir input unless it's a manifest or similar
            typer.secho(
                "Warning: --output is ignored when --dir is used without --output-dir. Files will be processed individually.",
                fg=typer.colors.YELLOW,
            )
        if json_output:
            typer.secho(
                "Error: --json output is not supported with --dir. Output is written to files in --output-dir or alongside originals.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        if output_trace:
            typer.secho(
                "Warning: --output-trace is ignored when --dir is used. Traces are not generated per file in directory mode.",
                fg=typer.colors.YELLOW,
            )
            # output_trace = None # Disable it
    else:  # Single text or file input, not --memory-path, not --dir
        if output_dir:
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
                "Info: No --output-dir specified for --dir input. Compressed files will be placed alongside originals with '_compressed' suffix.",
                fg=typer.colors.BLUE,
            )

    # --- Execution Logic ---
    if memory_path_arg:
        resolved_memory_path = Path(memory_path_arg).expanduser().resolve()
        try:
            persistent_store_engine = load_engine(resolved_memory_path) # This is the target store
        except FileNotFoundError:
            typer.secho(
                f"Error: Engine store at '{resolved_memory_path}' not found. Initialize with 'engine init'.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(
                f"Error loading engine store from '{resolved_memory_path}': {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # The 'engine_instance' (configured with LLM if needed) is used for one-shot compression
        if text is not None:
            actual_text = sys.stdin.read() if text == "-" else text
            _compress_text_to_memory(
                persistent_store_engine,
                actual_text,
                engine_instance, # one-shot compressor
                budget,
                verbose_stats,
                tokenizer_func,
            )
        elif file is not None:
            _compress_file_to_memory(
                persistent_store_engine,
                file,
                engine_instance, # one-shot compressor
                budget,
                verbose_stats,
                tokenizer_func,
            )
        elif dir_path is not None:
            _compress_directory_to_memory(
                persistent_store_engine,
                dir_path,
                engine_instance, # one-shot compressor
                budget,
                recursive,
                pattern,
                verbose_stats,
                tokenizer_func,
            )

        # Persist changes to the persistent_store_engine
        try:
            persistent_store_engine.save(resolved_memory_path)
            typer.secho(
                f"Content compressed (using '{final_engine_id}') and saved to engine store at '{resolved_memory_path}'.",
                fg=typer.colors.GREEN,
            )
        except Exception as e:
            typer.secho(
                f"Error saving engine store at '{resolved_memory_path}' after compression: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    else:  # Output to file or stdout using the configured engine_instance
        if text is not None:
            actual_text = sys.stdin.read() if text == "-" else text
            _compress_text_to_file_or_stdout(
                actual_text,
                engine_instance, # Pass instantiated engine
                budget,
                output_file,
                output_trace,
                verbose_stats,
                tokenizer_func,
                json_output,
            )
        elif file is not None:
            _compress_file_to_file_or_stdout(
                file,
                engine_instance, # Pass instantiated engine
                budget,
                output_file,
                output_trace,
                verbose_stats,
                tokenizer_func,
                json_output,
            )
        elif dir_path is not None:
            _compress_directory_to_files(
                dir_path,
                engine_instance, # Pass instantiated engine
                budget,
                output_dir,
                recursive,
                pattern,
                verbose_stats,
                tokenizer_func,
            )


# --- Helper Functions ---
# _get_one_shot_compression_engine is removed as its logic is now integrated into compress_command.

def _compress_text_core(
    text_content: str, engine: BaseCompressionEngine, budget: int, tokenizer: Any
) -> tuple[CompressedMemory, Optional[CompressionTrace], float]:
    """Core logic to compress text using a pre-configured engine instance."""
    # engine is already instantiated and passed in.
    start_time = time.time()
    result = engine.compress(text_content, budget, tokenizer=tokenizer)
    elapsed_ms = (time.time() - start_time) * 1000

    if isinstance(result, tuple) and len(result) == 2:
        compressed_mem, trace_obj = result
    elif isinstance(result, CompressedMemory):
        compressed_mem, trace_obj = result, None
    else:
        # Try to get engine name from its config for a more informative error
        engine_name = "current engine"
        if hasattr(engine, 'config') and isinstance(engine.config, dict) and 'name' in engine.config:
            engine_name = engine.config['name']
        elif hasattr(engine, 'id'): # if it has an id attribute
             engine_name = engine.id
        typer.secho(
            f"Error: Compression engine '{engine_name}' returned an unexpected result type: {type(result)}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if trace_obj and trace_obj.processing_ms is None:
        trace_obj.processing_ms = elapsed_ms

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
    engine: BaseCompressionEngine, # Changed from engine_id
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    json_output: bool,
) -> None:
    compressed_mem, trace_obj, elapsed_ms = _compress_text_core(
        text_content, engine, budget, tokenizer # Pass engine instance
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
    engine: BaseCompressionEngine, # Changed from engine_id
    budget: int,
    output_file: Optional[Path],
    trace_file: Optional[Path],
    verbose_stats: bool,
    tokenizer: Any,
    json_output: bool,
) -> None:
    try:
        text_content = file_path.read_text()
    except Exception as e:
        typer.secho(
            f"Error reading file {file_path}: {e}", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(1)

    compressed_mem, trace_obj, elapsed_ms = _compress_text_core(
        text_content, engine, budget, tokenizer # Pass engine instance
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
    engine: BaseCompressionEngine, # Changed from engine_id
    budget: int,
    output_dir_obj: Optional[Path],
    recursive: bool,
    pattern: str,
    verbose_stats: bool,
    tokenizer: Any,
) -> None:
    files_to_process = list(
        dir_path_obj.rglob(pattern) if recursive else dir_path_obj.glob(pattern)
    )
    if not files_to_process:
        typer.echo(f"No files matching pattern '{pattern}' found in '{dir_path_obj}'.")
        return

    processed_count = 0
    for input_file_path in files_to_process:
        if not input_file_path.is_file():
            continue

        typer.echo(f"Processing {input_file_path}...")
        try:
            text_content = input_file_path.read_text()
        except Exception as e:
            typer.secho(
                f"  Error reading file {input_file_path}: {e}",
                err=True,
                fg=typer.colors.YELLOW,
            )
            continue

        compressed_mem, trace_obj, elapsed_ms = _compress_text_core(
            text_content, engine, budget, tokenizer # Pass engine instance
        )

        # Determine output path for this file
        current_output_file_path = None  # Renamed current_output_file
        if output_dir_obj:
            try:
                output_dir_obj.mkdir(parents=True, exist_ok=True)
                rel_path = input_file_path.relative_to(dir_path_obj)
                current_output_file_path = output_dir_obj / rel_path
                current_output_file_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure sub-output-dir exists
            except Exception as e:
                typer.secho(
                    f"  Error creating output directory structure for {input_file_path} in {output_dir_obj}: {e}",
                    err=True,
                    fg=typer.colors.YELLOW,
                )
                continue  # Skip this file
        else:  # Output alongside original with suffix
            current_output_file_path = input_file_path.with_name(
                f"{input_file_path.stem}_compressed{input_file_path.suffix}"
            )

        # Output this file (trace_file is None for dir mode, json_output is False for dir mode)
        _output_results(
            f"file: {input_file_path.name}",
            compressed_mem,
            trace_obj,
            elapsed_ms,
            current_output_file_path,
            None,
            verbose_stats,
            tokenizer,
            False,
        )
        processed_count += 1

    if (
        verbose_stats or processed_count > 0
    ):  # Print summary if any file was processed or verbose
        typer.echo(f"Finished processing directory. Processed {processed_count} files.")


# --- Helpers for --memory-path ---


def _compress_text_to_memory(
    persistent_store_engine: BaseCompressionEngine,
    text_to_compress: str,
    one_shot_compressor_engine: BaseCompressionEngine, # The --engine for this run
    budget: int,
    verbose_stats: bool,
    tokenizer: Any,
    source_document_id: Optional[str] = "text_input",
) -> None:
    # Use the one_shot_compressor_engine (which might be LLM-backed) to compress
    compressed_mem, _, elapsed_ms = _compress_text_core(
        text_to_compress, one_shot_compressor_engine, budget, tokenizer
    )

    # Ingest the result into the persistent_store_engine
    if isinstance(persistent_store_engine, PrototypeEngine):
        persistent_store_engine.add_memory(
            compressed_mem.text, source_document_id=source_document_id
        )
    else:
        persistent_store_engine.ingest(compressed_mem.text)

    if verbose_stats:
        orig_tokens = token_count(tokenizer, text_to_compress)
        comp_tokens = token_count(tokenizer, compressed_mem.text)
        compressor_name = "current compressor" # Fallback name
        if hasattr(one_shot_compressor_engine, 'config') and isinstance(one_shot_compressor_engine.config, dict) and 'name' in one_shot_compressor_engine.config:
            compressor_name = one_shot_compressor_engine.config['name']
        elif hasattr(one_shot_compressor_engine, 'id'):
             compressor_name = one_shot_compressor_engine.id

        typer.echo(f"Source: {source_document_id}")
        typer.echo(
            f"Original tokens: {orig_tokens}\nCompressed tokens (using '{compressor_name}'): {comp_tokens}\nTime (one-shot) ms: {elapsed_ms:.1f}"
        )
        typer.echo(
            f"Compressed text ingested into main engine store at '{persistent_store_engine.path if hasattr(persistent_store_engine, 'path') else 'memory store'}'."
        )


def _compress_file_to_memory(
    persistent_store_engine: BaseCompressionEngine,
    file_path_obj: Path,
    one_shot_compressor_engine: BaseCompressionEngine,
    budget: int,
    verbose_stats: bool,
    tokenizer: Any,
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
        persistent_store_engine,
        text_content,
        one_shot_compressor_engine,
        budget,
        verbose_stats,
        tokenizer,
        source_document_id=str(file_path_obj.name),
    )


def _compress_directory_to_memory(
    persistent_store_engine: BaseCompressionEngine,
    dir_path_obj: Path,
    one_shot_compressor_engine: BaseCompressionEngine,
    budget: int,
    recursive: bool,
    pattern: str,
    verbose_stats: bool,
    tokenizer: Any,
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
            persistent_store_engine,
            input_file_path,
            one_shot_compressor_engine,
            budget,
            verbose_stats,
            tokenizer,
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
# - `compact_memory.prototype_engine.PrototypeEngine` for isinstance check.
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
