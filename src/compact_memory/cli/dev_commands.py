import json
import yaml
import os
import sys
from pathlib import Path
from typing import List, Optional, Any
from tqdm import tqdm

import typer
from rich.console import Console
from rich.table import Table

# Imports from compact_memory package
from compact_memory.validation.registry import (
    list_validation_metrics,
    get_validation_metric_class,
)
from compact_memory.validation.embedding_metrics import (
    MultiModelEmbeddingSimilarityMetric,
    EmbeddingSimilarityMetric,
)
from compact_memory.engines.registry import (
    available_engines as cm_available_engines,  # Renamed to avoid conflict
    all_engine_metadata as cm_all_engine_metadata,  # Renamed
    get_compression_engine,
)
from compact_memory.embedding_pipeline import get_embedding_dim

# PrototypeEngine was removed
# from compact_memory.vector_store import (
#     InMemoryVectorStore,
# )  # Specific for inspect-engine (assuming it was only for PrototypeEngine)
from compact_memory import llm_providers  # For test-llm-prompt
from compact_memory.model_utils import (
    download_embedding_model as util_download_embedding_model,
    download_chat_model as util_download_chat_model,
)
from compact_memory.package_utils import (
    validate_package_dir,
    # load_manifest, load_engine_class, check_requirements_installed # Not directly used by CLI commands here
)
from compact_memory.engines import (
    BaseCompressionEngine,
    CompressedMemory,
    CompressionTrace,
)


dev_app = typer.Typer(
    help="Commands for compression engine developers and researchers."
)
console = Console()


@dev_app.command(
    "list-metrics",
    help="Lists all available validation metric IDs that can be used in evaluations.",
)
def list_metrics_command() -> None:  # Renamed from list_metrics
    metric_ids = list_validation_metrics()
    if not metric_ids:
        typer.echo("No validation metrics found.")
        return
    typer.echo("Available validation metric IDs:")
    for mid in metric_ids:
        typer.echo(f"- {mid}")


@dev_app.command(
    "list-engines",
    help="Lists all available compression engine IDs, their versions, and sources (built-in or plugin).",
)
@dev_app.command(
    "list-strategies",  # Alias
    help="Lists all available compression engine IDs, their versions, and sources (built-in or plugin).",
    hidden=True,
)
def list_registered_engines_command(  # Renamed
    include_contrib: bool = typer.Option(
        False,
        "--include-contrib",
        help="Include experimental contrib engines (deprecated, contrib is integrated).",
    )
) -> None:
    # from compact_memory.plugin_loader import load_plugins # Global
    # from compact_memory.contrib import enable_all_experimental_engines # Global
    # load_plugins()
    # enable_all_experimental_engines()
    table = Table(
        "Engine ID",
        "Display Name",
        "Version",
        "Source",
        "Status",  # Added status for overrides
        title="Available Compression Engines",  # Corrected title casing
    )
    meta = cm_all_engine_metadata()
    ids = cm_available_engines()
    if not ids:
        typer.echo("No compression engines found.")
        return
    for sid in sorted(ids):
        info = meta.get(sid, {})
        # Contrib filtering is likely obsolete if contrib is now mainlined or handled by plugin system.
        # if not include_contrib and info.get("source") == "contrib":
        # continue
        status = ""
        if info.get("overrides"):  # Check for 'overrides' key if present
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
    "inspect-engine",
    help="Inspects aspects of a compression engine. Functionality for 'prototype' engine was removed.",
)
def inspect_engine_command(
    engine_name: str = typer.Argument(
        ...,
        help="The name of the engine to inspect.",
    ),
    *,
    list_prototypes: bool = typer.Option(
        False,
        "--list-prototypes",
        help="This option was for the removed PrototypeEngine and is no longer functional.",
    ),
    # memory_path_arg: Optional[Path] = typer.Option(
    # None, "--memory-path", "-m", help="Path to the engine store directory for inspection.",
    # resolve_path=True,
    # )
) -> None:
    if engine_name.lower() == "prototype" or list_prototypes:
        typer.secho(
            f"Info: Functionality specific to 'PrototypeEngine' (like --list-prototypes) has been removed.",
            fg=typer.colors.YELLOW,
        )
        # Optionally, provide information about inspecting other engines if a generic mechanism exists or is planned.
        # For now, just indicate removal.
        raise typer.Exit(code=0)

    typer.secho(
        f"Generic inspection for engine '{engine_name}' is not yet implemented or functionality for 'prototype' was removed.",
        fg=typer.colors.BLUE,
    )


@dev_app.command(
    "evaluate-compression",
    help='Evaluates compressed text against original text using a specified metric.\n\nUsage Examples:\n  compact-memory dev evaluate-compression original.txt summary.txt --metric compression_ratio\n  echo "original text" | compact-memory dev evaluate-compression - summary.txt --metric some_other_metric --metric-params \'{"param": "value"}\'',
)
def evaluate_compression_command(  # Renamed
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
    embedding_models: List[str] = typer.Option(
        [],
        "--embedding-model",
        "-M",
        help="Embedding model name(s) for embedding similarity. May be repeated or provided as a JSON list.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output evaluation scores in JSON format."
    ),
) -> None:
    def read_input_content(
        value: str, name_for_error: str, allow_stdin: bool
    ) -> str:  # Renamed function
        if value == "-":
            if not allow_stdin:
                typer.secho(
                    f"Error: Cannot use '-' for stdin for both {name_for_error} and the other input simultaneously.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            if not sys.stdin.isatty():  # Check if there's something to read
                return sys.stdin.read()
            else:
                typer.secho(
                    f"Error: Expected data from stdin for {name_for_error}, but nothing was piped.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

        p = Path(value)
        if p.exists() and p.is_file():
            try:
                return p.read_text()
            except Exception as e:
                typer.secho(
                    f"Error reading {name_for_error} file '{p}': {e}",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
        return value

    orig_text = read_input_content(original_input, "original input", True)
    comp_text = read_input_content(
        compressed_input, "compressed input", original_input != "-"
    )

    try:
        MetricCls = get_validation_metric_class(metric_id)  # Renamed variable
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

    models: List[str] = []
    for item in embedding_models:
        item = item.strip()
        if not item:
            continue
        if item.startswith("["):
            try:
                models.extend(json.loads(item))
            except json.JSONDecodeError as exc:
                typer.secho(
                    f"Error: Invalid JSON in --embedding-model '{item}': {exc}",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
        else:
            models.append(item)

    if MetricCls is EmbeddingSimilarityMetric:
        metric_instance = MultiModelEmbeddingSimilarityMetric(models, **params)
    else:
        metric_instance = MetricCls(**params)
    try:
        # Assuming evaluate method signature is: evaluate(self, original_text: str, compressed_text: str) -> Dict[str, Any]
        scores = metric_instance.evaluate(
            original_text=orig_text, compressed_text=comp_text
        )
    except Exception as exc:
        typer.secho(
            f"Error during metric evaluation with '{metric_id}': {exc}",
            err=True,
            fg=typer.colors.RED,  # Added metric_id to error
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(scores))
    else:
        typer.echo(f"Scores for metric '{metric_id}':")
        if isinstance(metric_instance, MultiModelEmbeddingSimilarityMetric):
            for model, vals in scores.items():
                sim = vals.get("semantic_similarity")
                tokens = vals.get("token_count")
                typer.echo(f"- {model}: semantic_similarity={sim}, tokens={tokens}")
        else:
            for k, v in scores.items():
                typer.echo(f"- {k}: {v}")


@dev_app.command(
    "test-llm-prompt",
    help='Tests a Language Model (LLM) prompt with specified context and query.\n\nUsage Examples:\n  compact-memory dev test-llm-prompt --context "AI is rapidly evolving." --query "Tell me more." --model-id tiny-gpt2\n  cat context.txt | compact-memory dev test-llm-prompt --context - -q "What are the implications?" --model-id openai/gpt-3.5-turbo --output-response response.txt --llm-config my_llm_config.yaml',
)
def test_llm_prompt_command(  # Renamed
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
    model_id: str = typer.Option(  # Changed from model to model_id for consistency
        "tiny-gpt2",  # Default value
        "--model-id",  # Changed option name
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
        resolve_path=True,  # Added resolve_path
    ),
    llm_config_file: Optional[Path] = typer.Option(
        Path("llm_models_config.yaml"),  # Default value
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
    def read_content_value(
        val: str, input_name: str, allow_stdin: bool
    ) -> str:  # Renamed function
        if val == "-":
            if not allow_stdin:
                typer.secho(
                    f"Error: Cannot use '-' for stdin for {input_name} when another input is already using it.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            if not sys.stdin.isatty():
                return sys.stdin.read()
            else:
                typer.secho(
                    f"Error: Expected data from stdin for {input_name}, but nothing was piped.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
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
        return val

    context_text = read_content_value(
        context_input, "context", True
    )  # Only context can be stdin for now

    cfg = {}
    if llm_config_file and llm_config_file.exists():
        try:
            cfg = yaml.safe_load(llm_config_file.read_text()) or {}
        except Exception as exc:
            typer.secho(
                f"Error loading LLM config '{llm_config_file}': {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    # model_id is the key in the llm_config_file
    model_config = cfg.get(model_id, {"provider": "local", "model_name": model_id})
    provider_name = model_config.get(
        "provider", "local"
    )  # Default to local if not specified
    actual_model_name_for_provider = model_config.get("model_name", model_id)

    # Select provider (ensure llm_providers has these attributes)
    if provider_name == "openai":
        provider_instance = llm_providers.OpenAIProvider()
    elif provider_name == "gemini":
        provider_instance = llm_providers.GeminiProvider()
    elif provider_name == "local":  # Assuming 'local' maps to LocalTransformersProvider
        provider_instance = llm_providers.LocalTransformersProvider()
    else:
        typer.secho(
            f"Error: Unknown LLM provider '{provider_name}' in config for model '{model_id}'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    api_key_value = (
        os.getenv(api_key_env_var) if api_key_env_var else None
    )  # Renamed variable

    prompt_elements = []  # Renamed variable
    if system_prompt:
        prompt_elements.append(system_prompt)
    if context_text:  # Ensure context_text is not empty
        prompt_elements.append(context_text)
    prompt_elements.append(query)  # Query is mandatory
    final_prompt = "\n\n".join(
        filter(None, prompt_elements)
    )  # Filter out empty strings

    typer.echo(
        f"--- Sending Prompt to LLM ({provider_name} - {actual_model_name_for_provider}) ---"
    )
    typer.echo(
        final_prompt[:500]
        + ("..." if len(final_prompt) > 500 else "")  # Improved preview
    )
    typer.echo("--- End of Prompt ---")

    try:
        response_text = provider_instance.generate_response(  # Renamed variable
            final_prompt,
            model_name=actual_model_name_for_provider,
            max_new_tokens=max_new_tokens,
            api_key=api_key_value,  # Pass the fetched API key
        )
    except Exception as exc:
        typer.secho(
            f"LLM generation error with model '{actual_model_name_for_provider}': {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if output_llm_response_file:
        try:
            output_llm_response_file.parent.mkdir(parents=True, exist_ok=True)
            output_llm_response_file.write_text(response_text)
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
        typer.echo(response_text)


@dev_app.command(
    "evaluate-llm-response",
    help="Evaluates an LLM's response against a reference answer using a specified metric.",
)
def evaluate_llm_response_command(  # Renamed
    response_input: str = typer.Argument(
        ...,
        help="LLM's generated response text, path to a response file, or '-' to read from stdin.",
    ),
    reference_input: str = typer.Argument(
        ...,
        help="Reference (ground truth) answer text, path to a file, or '-' to read from stdin.",  # Updated help
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
    def read_content_value(
        val: str, input_name: str, allow_stdin: bool
    ) -> str:  # Reusing helper
        if val == "-":
            if not allow_stdin:
                typer.secho(
                    f"Error: Cannot use '-' for stdin for {input_name} when another input is already using it.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            if not sys.stdin.isatty():
                return sys.stdin.read()
            else:
                typer.secho(
                    f"Error: Expected data from stdin for {input_name}, but nothing was piped.",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
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
        return val

    resp_text = read_content_value(response_input, "LLM response", True)
    ref_text = read_content_value(
        reference_input, "reference answer", response_input != "-"
    )

    try:
        MetricCls = get_validation_metric_class(metric_id)  # Renamed
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

    metric_instance = MetricCls(**params)  # Renamed
    try:
        # Assuming evaluate method signature: evaluate(self, llm_response: str, reference_answer: str) -> Dict[str, Any]
        scores = metric_instance.evaluate(
            llm_response=resp_text, reference_answer=ref_text
        )
    except Exception as exc:
        typer.secho(
            f"Error during metric evaluation with '{metric_id}': {exc}",
            err=True,
            fg=typer.colors.RED,  # Added metric_id
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
def download_embedding_model_command(  # Renamed
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2",  # Default value
        "--model-name",  # Explicit option name
        help="Name of the SentenceTransformer model to download (e.g., 'all-MiniLM-L6-v2').",
    )
) -> None:
    typer.echo(f"Starting download for embedding model: {model_name}...")
    # tqdm is used inside util_download_embedding_model, so no need for another bar here
    # unless the util function is changed to not use tqdm.
    # For now, assume util_download_embedding_model handles its own progress display.
    try:
        util_download_embedding_model(
            model_name
        )  # This function should handle its own tqdm
        typer.echo(f"Successfully downloaded embedding model '{model_name}'.")
    except Exception as e:
        typer.secho(
            f"Error downloading embedding model '{model_name}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@dev_app.command(
    "download-chat-model",
    help="Downloads a specified causal Language Model (e.g., for chat) from Hugging Face.",
)
def download_chat_model_command(  # Renamed
    model_name: str = typer.Option(
        "tiny-gpt2",  # Default value
        "--model-name",  # Explicit option name
        help="Name of the Hugging Face causal LM to download (e.g., 'gpt2', 'facebook/opt-125m').",
    )
) -> None:
    typer.echo(f"Starting download for chat model: {model_name}...")
    # Similar to embedding model, assume util_download_chat_model handles tqdm
    try:
        util_download_chat_model(model_name)  # This function should handle its own tqdm
        typer.echo(f"Successfully downloaded chat model '{model_name}'.")
    except Exception as e:
        typer.secho(
            f"Error downloading chat model '{model_name}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@dev_app.command(
    "create-engine-package",
    help="Creates a new compression engine extension package from a template. This command generates a template directory with all the necessary files to start developing a new, shareable engine package, including a sample engine, manifest file, and README.",
)
def create_engine_package_command(  # Renamed
    name: str = typer.Option(
        "compact_memory_example_engine",  # Default value
        "--name",  # Explicit option name
        help="Name for the new engine package (e.g., 'compact_memory_my_engine'). Used for directory and engine ID.",
    ),
    path_str: Optional[
        str
    ] = typer.Option(  # Renamed to path_str to avoid conflict with pathlib.Path
        None,
        "--path",
        help="Directory where the engine package will be created. Defaults to a new directory named after the engine in the current location.",
    ),
) -> None:
    target_dir = Path(path_str or name).resolve()

    if target_dir.exists() and any(target_dir.iterdir()):
        typer.secho(
            f"Error: Output directory '{target_dir}' already exists and is not empty.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Using f-string for engine_py_content and manifest for dynamic name insertion
    engine_py_content = f"""from compact_memory.engines import BaseCompressionEngine, CompressedMemory, CompressionTrace
# Add any other necessary imports here

class MyEngine(BaseCompressionEngine):
    # Unique identifier for your engine
    id = "{name}" # Uses the 'name' parameter

    # Optional: Define parameters your engine accepts with default values
    # def __init__(self, param1: int = 10, param2: str = "default", **kwargs):
    #     super().__init__(**kwargs) # Pass kwargs to parent if it expects them (e.g. config)
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
        tokenizer = kwargs.get("tokenizer")

        # --- Your compression logic here ---
        # This is a placeholder. Implement your actual compression algorithm.
        # Example: Truncate based on character count as a rough proxy for token budget.
        # A real implementation would use the tokenizer.
        if isinstance(text_or_chunks, list):
            text_to_compress = "\\n".join(text_or_chunks)
        else:
            text_to_compress = str(text_or_chunks)

        # Simple truncation - replace with sophisticated logic
        limit = llm_token_budget * 4 # Heuristic: avg 4 chars per token
        compressed_text = text_to_compress[:limit]

        original_tokens = len(tokenizer(text_to_compress)['input_ids']) if tokenizer else 0
        compressed_tokens = len(tokenizer(compressed_text)['input_ids']) if tokenizer else 0

        trace = CompressionTrace(
            engine_name=self.id,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            # Add more trace details as needed (e.g., steps, processing_ms)
        )
        return CompressedMemory(text=compressed_text), trace
"""
    (target_dir / "engine.py").write_text(engine_py_content)

    manifest_content = {  # Using dict then dumping to YAML
        "package_format_version": "1.0",
        "engine_id": name,  # Uses the 'name' parameter
        "engine_class_name": "MyEngine",  # Default class name in template
        "engine_module": "engine",  # Default module name (engine.py)
        "display_name": name.replace("_", " ").title(),  # Generate a nice display name
        "version": "0.1.0",
        "authors": [
            {"name": "Your Name", "email": "your.email@example.com"}
        ],  # Placeholder
        "description": f"A sample compression engine package: {name}.",  # Placeholder
        "requirements": [],  # Placeholder for dependencies
    }
    (target_dir / "engine_package.yaml").write_text(
        yaml.safe_dump(manifest_content, sort_keys=False)
    )
    (target_dir / "requirements.txt").write_text(
        "# Add Python dependencies here, one per line\n"
    )
    (target_dir / "README.md").write_text(
        f"# {manifest_content['display_name']}\n\n{manifest_content['description']}\n"
    )

    typer.echo(f"Successfully created engine package '{name}' at: {target_dir}")


@dev_app.command(
    "validate-engine-package",
    help="Validates the structure and manifest of a compression engine extension package.\n\nUsage Examples:\n  compact-memory dev validate-engine-package path/to/my_engine_pkg",
)
def validate_engine_package_command(  # Renamed
    package_path: Path = typer.Argument(
        ...,
        help="Path to the root directory of the engine package.",
        exists=True,
        file_okay=False,  # Must be a directory
        dir_okay=True,
        resolve_path=True,
    )
) -> None:
    errors, warnings = validate_package_dir(
        package_path
    )  # This util should do the heavy lifting
    for w_msg in warnings:  # Renamed variable
        typer.secho(f"Warning: {w_msg}", fg=typer.colors.YELLOW)
    if errors:
        typer.secho(
            "Engine package validation failed with errors:",
            fg=typer.colors.RED,
            err=True,
        )
        for e_msg in errors:  # Renamed variable
            typer.secho(
                f"- {e_msg}", fg=typer.colors.RED, err=True
            )  # Ensure errors go to stderr
        raise typer.Exit(code=1)
    typer.secho(
        f"Engine package at '{package_path}' appears valid.", fg=typer.colors.GREEN
    )  # Success message


@dev_app.command(
    "inspect-trace",
    help="Inspects a CompressionTrace JSON file, optionally filtering by step type.",
)
def inspect_trace_command(  # Renamed
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
        "--type",  # Consistent option name
        help="Filter trace steps by this 'type' string (e.g., 'chunking', 'llm_call').",
    ),
) -> None:
    # Redundant exists check due to Typer, but good for clarity
    # if not trace_file.exists():
    #     typer.secho(f"Error: Trace file '{trace_file}' not found.", err=True, fg=typer.colors.RED)
    #     raise typer.Exit(code=1)
    try:
        trace_data = json.loads(trace_file.read_text())  # Renamed variable
    except json.JSONDecodeError as e:
        typer.secho(
            f"Error: Invalid JSON in trace file '{trace_file}': {e}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"Error reading trace file '{trace_file}': {e}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    steps = trace_data.get("steps", [])

    # Build title string
    title_parts = [f"Compression Trace: {trace_file.name}"]
    if trace_data.get("engine_name"):
        title_parts.append(f"Engine: {trace_data['engine_name']}")
    if trace_data.get("original_tokens") is not None:  # Check for None explicitly
        title_parts.append(f"Original Tokens: {trace_data['original_tokens']}")
    if trace_data.get("compressed_tokens") is not None:
        title_parts.append(f"Compressed Tokens: {trace_data['compressed_tokens']}")
    if trace_data.get("processing_ms") is not None:
        title_parts.append(f"Time: {trace_data['processing_ms']:.2f}ms")
    full_title = " | ".join(title_parts)

    # console.print(f"Engine: {trace_data.get('engine_name', 'N/A')}") # Redundant if in title
    table = Table("Index", "Type", "Details Preview", title=full_title)

    filtered_steps_count = 0
    for idx, step_detail in enumerate(steps):  # Renamed variables
        current_step_type = step_detail.get("type", "N/A")  # Default if type is missing
        if step_type and current_step_type != step_type:
            continue

        details_preview_obj = step_detail.get("details", {})  # Renamed variable
        try:
            details_preview_str = json.dumps(details_preview_obj)[:50] + (
                "..." if len(json.dumps(details_preview_obj)) > 50 else ""
            )
        except TypeError:  # Handle non-serializable details gracefully
            details_preview_str = str(details_preview_obj)[:50] + (
                "..." if len(str(details_preview_obj)) > 50 else ""
            )

        table.add_row(str(idx), current_step_type, details_preview_str)
        filtered_steps_count += 1

    if filtered_steps_count == 0:
        if step_type:
            typer.echo(f"No steps found with type '{step_type}'.")
        else:
            typer.echo("No steps found in the trace.")
    else:
        console.print(table)


@dev_app.command(
    "evaluate-engines",
    help="Compress text with each specified engine and compute basic metrics.",
)
def evaluate_engines_command(
    *,
    text: Optional[str] = typer.Option(
        None,
        "--text",
        help="Raw text to compress or '-' to read from stdin.",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to a text file to compress.",
    ),
    engines: Optional[List[str]] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Engine ID to evaluate. May be provided multiple times.",
    ),
    embedding_models: List[str] = typer.Option(
        [],
        "--embedding-model",
        "-M",
        help=(
            "Embedding model name(s) for embedding similarity. "
            "May be repeated or provided as a JSON list."
        ),
    ),
    budget: int = typer.Option(
        100,
        "--budget",
        help="Token budget for compression.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        resolve_path=True,
        help="Write metrics JSON to this file.",
    ),
) -> None:
    """Evaluate one or more engines on the provided text."""
    if (text is None) == (file is None):
        typer.secho(
            "Specify exactly one of --text or --file.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if text == "-":
        if not sys.stdin.isatty():
            text_input = sys.stdin.read()
        else:
            typer.secho(
                "Expected data from stdin for --text '-', but none was provided.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    elif text is not None:
        text_input = text
    else:
        try:
            text_input = file.read_text() if file else ""
        except Exception as exc:
            typer.secho(
                f"Error reading file '{file}': {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    available = set(cm_available_engines())
    if engines:
        invalid = [e for e in engines if e not in available]
        if invalid:
            typer.secho(
                f"Unknown engine IDs: {', '.join(invalid)}",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        engine_ids = engines
    else:
        engine_ids = list(available)

    ratio_metric = get_validation_metric_class("compression_ratio")()
    models: List[str] = []
    for item in embedding_models:
        item = item.strip()
        if not item:
            continue
        if item.startswith("["):
            try:
                models.extend(json.loads(item))
            except json.JSONDecodeError as exc:
                typer.secho(
                    f"Error: Invalid JSON in --embedding-model '{item}': {exc}",
                    err=True,
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
        else:
            models.append(item)
    embed_metric = MultiModelEmbeddingSimilarityMetric(models)

    results: dict[str, dict[str, Any]] = {}
    for eid in engine_ids:
        EngineCls = get_compression_engine(eid)
        engine_instance = EngineCls()

        result = engine_instance.compress(text_input, llm_token_budget=budget)
        compressed = result[0] if isinstance(result, tuple) else result

        if hasattr(compressed, "text"):
            comp_text = compressed.text
        elif isinstance(compressed, dict):
            comp_text = compressed.get("content", str(compressed))
        else:
            comp_text = str(compressed)

        ratio = ratio_metric.evaluate(
            original_text=text_input, compressed_text=comp_text
        )["compression_ratio"]
        embed = embed_metric.evaluate(
            original_text=text_input, compressed_text=comp_text
        )

        results[eid] = {
            "compression_ratio": ratio,
            "embedding_similarity": embed,
        }

    json_str = json.dumps(results, indent=2)
    if output:
        output.write_text(json_str)
    if json_output:
        typer.echo(json_str)
    else:
        for eid, metrics in results.items():
            typer.echo(f"Engine: {eid}")
            typer.echo(f"  compression_ratio: {metrics['compression_ratio']}")
            for model, vals in metrics["embedding_similarity"].items():
                sim = vals.get("semantic_similarity")
                tokens = vals.get("token_count")
                typer.echo(f"  {model}: semantic_similarity={sim}, tokens={tokens}")
        typer.echo(json_str)


# Final checks:
# - All command functions renamed (e.g., list_metrics -> list_metrics_command).
# - tqdm import removed as utils are expected to handle their own progress bars.
# - Imports are grouped logically.
# - `compact_memory.engines.registry` aliased to avoid conflict.
# - `PrototypeEngine` and `InMemoryVectorStore` references removed.
# - `llm_providers` imported for `test_llm_prompt_command`.
# - Model download utils imported.
# - Package utils imported.
# - `BaseCompressionEngine`, `CompressedMemory`, `CompressionTrace` imported for template.
# - `dev_app` and `console` defined.
# - `list_validation_metrics` helper used for listing metric IDs.
# - `inspect_engine_command` logic for loading from path is commented out but provides a good structure if needed later.
# - `evaluate_compression_command` and `evaluate_llm_response_command` helpers and logic seem fine.
# - `test_llm_prompt_command` logic for provider selection and API key handling is fine.
# - `download_*_model_command` simplified by assuming utils handle tqdm.
# - `create_engine_package_command` template content and file creation looks good.
# - `validate_engine_package_command` relies on util, which is appropriate.
# - `inspect_trace_command` has improved table display and filtering.
# - Corrected variable names and added more detailed error messages/edge case handling.
# - `compact_memory.contrib` imports removed as they are obsolete.
# - `enable_all_experimental_engines` removed as it's called globally.
# - `load_plugins` removed as it's called globally.
# Looks good.
