import json
import shutil
import os
import yaml
from pathlib import Path
from typing import Optional, Any, Tuple, Dict # Added Tuple, Dict
import logging
import time
from dataclasses import asdict
import sys
from tqdm import tqdm
import runpy
import functools # For functools.partial
import argparse # For Namespace in type hints if used directly

import typer
import portalocker
from .token_utils import token_count
from rich.table import Table
from rich.console import Console

from compact_memory import __version__
from .logging_utils import configure_logging


from .agent import Agent
from .vector_store import BaseVectorStore, InMemoryVectorStore # Added BaseVectorStore
from .vector_stores.chroma_adapter import ChromaVectorStoreAdapter # Added
from .vector_stores.faiss_adapter import FaissVectorStoreAdapter # Added

from . import local_llm
from .active_memory_manager import ActiveMemoryManager
from .registry import (
    _VALIDATION_METRIC_REGISTRY,
    get_validation_metric_class,
)
from . import llm_providers
from .model_utils import (
    download_embedding_model as util_download_embedding_model,
    download_chat_model as util_download_chat_model,
)
from .embedding_pipeline import (
    # get_embedding_dim, # This is now HF specific, use get_embedding_dim_hf directly
    EmbeddingDimensionMismatchError,
    EmbeddingFunction, # Added
    MockEncoder, # Added
    embed_text # Main embed_text to pass embedding_fn to Agent
)
from .embedding_providers.huggingface import ( # Added
    embed_text_hf,
    get_embedding_dim_hf,
    DEFAULT_MODEL_NAME as HF_DEFAULT_MODEL_NAME,
    DEFAULT_DEVICE as HF_DEFAULT_DEVICE
)
# from .utils import load_agent # This old load_agent is replaced by Agent.load_agent and CLI logic
from compact_memory.config import Config, DEFAULT_CONFIG, USER_CONFIG_PATH

from .compression import (
    available_strategies,
    get_compression_strategy,
    all_strategy_metadata,
    get_strategy_metadata,
)
from .plugin_loader import load_plugins
from .cli_plugins import load_cli_plugins
from .package_utils import (
    load_manifest,
    validate_manifest,
    load_strategy_class,
    validate_package_dir,
    check_requirements_installed,
)
from .response_experiment import ResponseExperimentConfig, run_response_experiment
from .experiment_runner import (
    ExperimentConfig,
    run_experiment,
)

app = typer.Typer(
    help="Compact Memory: A CLI for intelligent information management using memory agents and advanced compression. Ingest, query, and compress information. Manage agent configurations and developer tools."
)
console = Console()

# --- New Command Groups ---
agent_app = typer.Typer(
    help="Manage memory agents: initialize, inspect statistics, validate, and clear."
)
config_app = typer.Typer(
    help="Manage Compact Memory application configuration settings."
)
dev_app = typer.Typer(
    help="Commands for compression strategy developers and researchers."
)

# --- Add New Command Groups to Main App ---
app.add_typer(agent_app, name="agent")
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
        help="Path to the Compact Memory agent directory. Overrides COMPACT_MEMORY_PATH env var and configuration files.",
    ),
    model_id: Optional[str] = typer.Option( # This is for LLM model, not embedding model
        None,
        "--model-id",
        help="Default model ID for LLM interactions. Overrides COMPACT_MEMORY_DEFAULT_MODEL_ID env var and configuration files.",
    ),
    strategy_id: Optional[str] = typer.Option(
        None,
        "--strategy-id",
        help="Default compression strategy ID. Overrides COMPACT_MEMORY_DEFAULT_STRATEGY_ID env var and configuration files.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the application version and exit.",
    ),
    # New global options for vector store and embeddings (can be overridden by subcommands)
    vector_store_type: Optional[str] = typer.Option(
        None, "--vector-store-type", help="Vector store type (memory, chroma, faiss)."
    ),
    vector_store_path: Optional[str] = typer.Option(
        None, "--vector-store-path", help="Path for persistent vector stores."
    ),
    vector_store_collection: Optional[str] = typer.Option(
        None, "--vector-store-collection", help="Collection name for Chroma-like stores."
    ),
    embedding_provider_type: Optional[str] = typer.Option(
        None, "--embedding-provider-type", help="Embedding provider (huggingface, mock)."
    ),
    embedding_model_name: Optional[str] = typer.Option(
        None, "--embedding-model-name", help="Name of the embedding model."
    ),
    embedding_device: Optional[str] = typer.Option(
        None, "--embedding-device", help="Device for embeddings (cpu, cuda)."
    ),
    embedding_dim: Optional[int] = typer.Option(
        None, "--embedding-dim", help="Embedding dimension (required for some custom providers)."
    ),
) -> None:
    if ctx.obj is None:
        ctx.obj = {}
    if "config" not in ctx.obj:
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

    # Prioritize CLI args for these new global settings, then config, then defaults
    # These will be stored in ctx.obj and can be accessed by subcommands
    # However, the _create_dependencies_from_config will handle the hierarchy more directly.
    ctx.obj["cli_vector_store_type"] = vector_store_type
    ctx.obj["cli_vector_store_path"] = vector_store_path
    ctx.obj["cli_vector_store_collection"] = vector_store_collection
    ctx.obj["cli_embedding_provider_type"] = embedding_provider_type
    ctx.obj["cli_embedding_model_name"] = embedding_model_name
    ctx.obj["cli_embedding_device"] = embedding_device
    ctx.obj["cli_embedding_dim"] = embedding_dim

    resolved_memory_path = (
        memory_path if memory_path is not None else config.get("compact_memory_path")
    )
    resolved_model_id = (
        model_id if model_id is not None else config.get("default_model_id")
    )
    resolved_strategy_id = (
        strategy_id if strategy_id is not None else config.get("default_strategy_id")
    )

    if resolved_memory_path:
        resolved_memory_path = str(Path(resolved_memory_path).expanduser())

    command_requires_memory_path = True
    if ctx.invoked_subcommand:
        no_mem_path_commands = ["config", "dev"]
        if ctx.invoked_subcommand in no_mem_path_commands:
            command_requires_memory_path = False
        else:
            parent_command_parts = ctx.invoked_subcommand.split(".")
            if parent_command_parts and parent_command_parts[0] in no_mem_path_commands:
                 command_requires_memory_path = False


    if command_requires_memory_path and not resolved_memory_path:
        # ... (rest of memory path handling) ...
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


    load_plugins()
    ctx.obj.update(
        {
            "verbose": resolved_verbose,
            "log_file": resolved_log_file,
            "compact_memory_path": resolved_memory_path,
            "default_model_id": resolved_model_id,
            "default_strategy_id": resolved_strategy_id,
        }
    )

# --- PersistenceLock and _corrupt_exit remain unchanged ---
class PersistenceLock:
    def __init__(self, path: Path) -> None:
        self.file = (path / ".lock").open("a+")

    def __enter__(self):
        portalocker.lock(self.file, portalocker.LockFlags.EXCLUSIVE)
        return self

    def __exit__(self, exc_type, exc, tb):
        portalocker.unlock(self.file)
        self.file.close()


def _corrupt_exit(path: Path, exc: Exception) -> None:
    typer.echo(f"Error: Brain data is corrupted. {exc}", err=True)
    typer.echo(
        f"Try running compact-memory validate {path} for more details or restore from a backup.",
        err=True,
    )
    raise typer.Exit(code=1)

# --- Helper for creating VS and EF ---
def _create_dependencies_from_config(
    agent_dir_path: Optional[Path], # Path to agent dir to load existing config
    # CLI arguments to override persisted config or for new agent
    cli_vs_type: Optional[str],
    cli_vs_path: Optional[str],
    cli_vs_collection: Optional[str],
    cli_emb_provider: Optional[str],
    cli_emb_model: Optional[str],
    cli_emb_device: Optional[str],
    cli_emb_dim: Optional[int],
) -> Tuple[BaseVectorStore, Optional[EmbeddingFunction], int]:

    agent_config: Dict[str, Any] = {}
    if agent_dir_path and (agent_dir_path / "agent_config.json").exists():
        with open(agent_dir_path / "agent_config.json", "r") as f:
            agent_config = json.load(f)

    # Determine Embedding Dimension
    final_embedding_dim = Agent.DEFAULT_EMBEDDING_DIM # Fallback default
    if cli_emb_dim is not None:
        final_embedding_dim = cli_emb_dim
    elif agent_config.get("embedding_dim") is not None:
        final_embedding_dim = agent_config["embedding_dim"]
    # If using default HF and no dim set, it will be inferred below.

    # Determine Embedding Function
    embedding_provider_type = cli_emb_provider or agent_config.get("embedding_provider_type", "huggingface")
    embedding_fn: Optional[EmbeddingFunction] = None

    if embedding_provider_type == "huggingface":
        model_name = cli_emb_model or agent_config.get("embedding_model_name", HF_DEFAULT_MODEL_NAME)
        device = cli_emb_device or agent_config.get("embedding_device", HF_DEFAULT_DEVICE)
        # embedding_fn = functools.partial(embed_text_hf, model_name=model_name, device=device)
        # Let embed_text handle defaulting if embedding_fn is None, simplifies Agent constructor
        embedding_fn = None # Agent will use its defaults or this can be set to the partial
        if final_embedding_dim == Agent.DEFAULT_EMBEDDING_DIM and cli_emb_dim is None and agent_config.get("embedding_dim") is None: # only infer if not explicitly set
            try:
                final_embedding_dim = get_embedding_dim_hf(model_name=model_name, device=device)
            except Exception as e:
                logging.warning(f"Could not infer embedding dimension for {model_name}: {e}. Using default {final_embedding_dim}.")
    elif embedding_provider_type == "mock":
        mock_encoder = MockEncoder()
        embedding_fn = mock_encoder.encode
        if final_embedding_dim == Agent.DEFAULT_EMBEDDING_DIM and cli_emb_dim is None and agent_config.get("embedding_dim") is None:
            final_embedding_dim = mock_encoder.dim
        elif final_embedding_dim != mock_encoder.dim:
             typer.secho(f"Warning: Provided embedding_dim {final_embedding_dim} does not match MockEncoder dim {mock_encoder.dim}. Using {mock_encoder.dim}.", fg=typer.colors.YELLOW)
             final_embedding_dim = mock_encoder.dim
    else: # Custom or None
        embedding_fn = None # Agent will use its default HF if this is None
        if final_embedding_dim == Agent.DEFAULT_EMBEDDING_DIM: # if still default
             typer.secho("Warning: Using default embedding dimension. Provide --embedding-dim if using a custom non-HF embedder.", fg=typer.colors.YELLOW)


    # Determine Vector Store
    vector_store_type = cli_vs_type or agent_config.get("vector_store_type", "memory")
    vs_path_str = cli_vs_path or agent_config.get("vector_store_config", {}).get("path")
    vs_collection = cli_vs_collection or agent_config.get("vector_store_config", {}).get("collection_name", "compact_memory_default")

    vector_store: BaseVectorStore
    using_default_vs_and_ef = True

    if vector_store_type == "chroma":
        if not ChromaVectorStoreAdapter: raise ImportError("ChromaDB support not fully installed.")
        vector_store = ChromaVectorStoreAdapter(
            path=vs_path_str, # Can be None for ephemeral
            collection_name=vs_collection,
            # embedding_function is not set here as Agent passes precomputed vectors
        )
        logging.info(f"Using ChromaVectorStoreAdapter (path: {vs_path_str}, collection: {vs_collection})")
        using_default_vs_and_ef = False
    elif vector_store_type == "faiss":
        if not FaissVectorStoreAdapter: raise ImportError("FAISS support not fully installed.")
        # Path for Faiss is handled by its load/persist methods, not directly in constructor for existing.
        # For new Faiss, it's in-memory until persisted.
        vector_store = FaissVectorStoreAdapter(embedding_dim=final_embedding_dim)
        logging.info(f"Using FaissVectorStoreAdapter (embedding_dim: {final_embedding_dim})")
        # Path will be used during agent.load_agent -> vs.load(path_from_config)
        using_default_vs_and_ef = False
    elif vector_store_type == "memory":
        vector_store = InMemoryVectorStore(embedding_dim=final_embedding_dim)
        logging.info(f"Using InMemoryVectorStore (embedding_dim: {final_embedding_dim})")
    else:
        typer.secho(f"Unsupported vector store type: {vector_store_type}. Defaulting to InMemoryVectorStore.", fg=typer.colors.YELLOW)
        vector_store = InMemoryVectorStore(embedding_dim=final_embedding_dim)

    if using_default_vs_and_ef and embedding_provider_type == "huggingface" and vector_store_type == "memory":
         if agent_dir_path is None: # Only for new agent init
            typer.echo("Using default in-memory vector store and local MiniLM embedder.")

    return vector_store, embedding_fn, final_embedding_dim


# --- Agent Commands ---
@agent_app.command(
    "init",
    help="Initializes a new Compact Memory agent in a specified directory.",
)
def init_agent_command( # Renamed from init
    ctx: typer.Context,
    target_directory: Path = typer.Argument(
        ...,
        help="Directory to initialize the new agent in. Will be created if it doesn't exist.",
        resolve_path=True,
    ),
    name: str = typer.Option("default", help="A descriptive name for the agent (stored in config)."),
    # tau and alpha are PSS/Agent configs, not direct VS/EF configs, but can be set at init
    tau: float = typer.Option(0.8, help="Similarity threshold (tau) for memory consolidation."),
    # CLI options for VS and EF for init
    vector_store_type: str = typer.Option("memory", "--vector-store-type", help="Vector store type (memory, chroma, faiss)."),
    vector_store_path: Optional[str] = typer.Option(None, "--vector-store-path", help="Path for persistent vector stores."),
    vector_store_collection: Optional[str] = typer.Option("compact_memory_default", "--vector-store-collection", help="Collection name for Chroma."),
    embedding_provider_type: str = typer.Option("huggingface", "--embedding-provider-type", help="Embedding provider (huggingface, mock)."),
    embedding_model_name: Optional[str] = typer.Option(None, "--embedding-model-name", help="Name of the embedding model."), # Default handled by helper
    embedding_device: Optional[str] = typer.Option(None, "--embedding-device", help="Device for embeddings (cpu, cuda)."), # Default handled by helper
    embedding_dim: Optional[int] = typer.Option(None, "--embedding-dim", help="Embedding dimension."),
) -> None:
    agent_path = target_directory.expanduser()
    if agent_path.exists() and any(agent_path.iterdir()):
        typer.secho(f"Error: Directory '{agent_path}' already exists and is not empty.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    agent_path.mkdir(parents=True, exist_ok=True)

    # For init, there's no existing agent_config.json to load, so helper uses CLI args or defaults.
    # Pass None for agent_dir_path to _create_dependencies_from_config for init
    try:
        vs_instance, efn_instance, emb_dim_resolved = _create_dependencies_from_config(
            agent_dir_path=None,
            cli_vs_type=vector_store_type,
            cli_vs_path=vector_store_path,
            cli_vs_collection=vector_store_collection,
            cli_emb_provider=embedding_provider_type,
            cli_emb_model=embedding_model_name,
            cli_emb_device=embedding_device,
            cli_emb_dim=embedding_dim
        )
    except ImportError as e:
        typer.secho(f"Error initializing dependencies: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Create agent instance
    # Other agent params like chunker, summary_creator can be added as CLI options too if needed
    agent = Agent(
        vector_store=vs_instance,
        embedding_dim=emb_dim_resolved,
        embedding_fn=efn_instance,
        similarity_threshold=tau,
        # Add other relevant params if they become CLI options for init
    )

    # Save the agent - this will write agent_config.json including VS and EF choices
    # The agent.save_agent() method needs to store these new config details.
    agent.save_agent(str(agent_path))

    # Store the agent name and other high-level CLI-provided metadata in a separate file or part of agent_config
    # For now, agent_config.json written by agent.save_agent() is the primary config.
    # If 'name' or 'alpha' needs to be stored, agent_config.json is the place.
    # agent.save_agent should be updated to include these if they are part of Agent's state/config.
    # Currently, agent_config saves embedding_dim, vs_type, vs_config, ef_provider, ef_model, ef_device.
    # Let's assume 'name' is more of a label for the directory or a meta field if needed.

    typer.echo(f"Successfully initialized Compact Memory agent at {agent_path}")
    typer.echo(f"  Vector Store: {type(vs_instance).__name__}")
    typer.echo(f"  Embedding Dim: {emb_dim_resolved}")
    if efn_instance:
        typer.echo(f"  Custom Embedding Fn: Yes")
    else:
        typer.echo(f"  Embedding Provider: {embedding_provider_type} (Model: {embedding_model_name or HF_DEFAULT_MODEL_NAME})")


# Simplified _load_agent_cli for internal CLI use, replacing the old global one from .utils
def _load_agent_cli(
    agent_dir_path_str: str,
    # CLI overrides for dependencies
    cli_vs_type: Optional[str],
    cli_vs_path: Optional[str],
    cli_vs_collection: Optional[str],
    cli_emb_provider: Optional[str],
    cli_emb_model: Optional[str],
    cli_emb_device: Optional[str],
    cli_emb_dim: Optional[int],
) -> Agent:
    agent_dir_path = Path(agent_dir_path_str)
    if not (agent_dir_path / "agent_config.json").exists():
        typer.secho(f"Error: Agent configuration not found at {agent_dir_path}. Has the agent been initialized?", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        vs_instance, efn_instance, emb_dim_resolved = _create_dependencies_from_config(
            agent_dir_path=agent_dir_path,
            cli_vs_type=cli_vs_type,
            cli_vs_path=cli_vs_path,
            cli_vs_collection=cli_vs_collection,
            cli_emb_provider=cli_emb_provider,
            cli_emb_model=cli_emb_model,
            cli_emb_device=cli_emb_device,
            cli_emb_dim=cli_emb_dim
        )
    except ImportError as e:
        typer.secho(f"Error initializing dependencies for loading agent: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e: # Catch other errors from helper
        typer.secho(f"Error creating dependencies: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


    # Agent.load_agent expects the vector_store_instance to be already loaded with its data
    # if it's a persistent store.
    # The _create_dependencies_from_config gives us the instance.
    # For stores like Faiss/Chroma that load from path, their .load() is called by Agent.load_agent.
    # This seems fine. Agent.load_agent gets the right type of store, and then populates it.

    try:
        # Pass the instantiated (but not necessarily data-loaded for persistable stores) VS and EF
        agent = Agent.load_agent(
            agent_dir_path=str(agent_dir_path),
            vector_store_instance=vs_instance, # This instance will be loaded by Agent.load_agent
            embedding_fn=efn_instance
            # embedding_dim is loaded from agent_config inside Agent.load_agent
            # chunker_class, summary_creator_class can be added if made configurable
        )
        return agent
    except EmbeddingDimensionMismatchError as exc: # Catch specific error from Agent.load_agent
        _corrupt_exit(agent_dir_path, exc)
    except FileNotFoundError as e: # If agent_config.json or other crucial files are missing
        typer.secho(f"Error loading agent: {e}. Ensure agent directory is correct and agent has been initialized.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as exc: # Catch other generic errors
        _corrupt_exit(agent_dir_path, exc) # type: ignore


@agent_app.command(
    "stats",
    help="Displays statistics about the Compact Memory agent.",
)
def stats_command( # Renamed from stats
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(
        None, "--memory-path", "-m", help="Path to the agent directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output statistics in JSON format."),
) -> None:
    agent_dir_str = memory_path_arg or ctx.obj.get("compact_memory_path")
    if not agent_dir_str: # Should be caught by main() if required by command
        typer.secho("Error: Memory path not specified.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    agent = _load_agent_cli(
        agent_dir_str,
        cli_vs_type=ctx.obj.get("cli_vector_store_type"),
        cli_vs_path=ctx.obj.get("cli_vector_store_path"),
        cli_vs_collection=ctx.obj.get("cli_vector_store_collection"),
        cli_emb_provider=ctx.obj.get("cli_embedding_provider_type"),
        cli_emb_model=ctx.obj.get("cli_embedding_model_name"),
        cli_emb_device=ctx.obj.get("cli_embedding_device"),
        cli_emb_dim=ctx.obj.get("cli_embedding_dim"),
    )
    data = agent.get_statistics()
    # ... (rest of stats display)
    logging.debug("Collected statistics: %s", data)
    if json_output:
        typer.echo(json.dumps(data))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


@agent_app.command("validate", help="Validates the integrity of the agent's storage.")
def validate_agent_storage_command( # Renamed
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option( None, "--memory-path", "-m", help="Path to the agent directory."),
) -> None:
    agent_dir_str = memory_path_arg or ctx.obj.get("compact_memory_path")
    if not agent_dir_str: typer.secho("Error: Memory path not specified.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    try:
        # Loading the agent itself is a form of validation
        _ = _load_agent_cli(
            agent_dir_str,
            cli_vs_type=ctx.obj.get("cli_vector_store_type"),
            cli_vs_path=ctx.obj.get("cli_vector_store_path"),
            cli_vs_collection=ctx.obj.get("cli_vector_store_collection"),
            cli_emb_provider=ctx.obj.get("cli_embedding_provider_type"),
            cli_emb_model=ctx.obj.get("cli_embedding_model_name"),
            cli_emb_device=ctx.obj.get("cli_embedding_device"),
            cli_emb_dim=ctx.obj.get("cli_embedding_dim"),
        )
        # TODO: Add more specific validation checks if needed, e.g., vector store consistency
        typer.echo(f"Agent storage at '{agent_dir_str}' appears valid based on successful load.")
    except typer.Exit: # If _load_agent_cli exited, re-raise to stop
        raise
    except Exception as exc:
        typer.secho(f"Error validating agent at '{agent_dir_str}': {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


@agent_app.command(
    "clear",
    help="Deletes all data from an agent's memory. This action is irreversible.",
)
def clear_command( # Renamed
    ctx: typer.Context,
    memory_path_arg: Optional[str] = typer.Option(None, "--memory-path", "-m", help="Path to the agent directory."),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without prompting."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate deletion."),
) -> None:
    agent_dir_str = memory_path_arg or ctx.obj.get("compact_memory_path")
    if not agent_dir_str: typer.secho("Error: Memory path not specified.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    path = Path(agent_dir_str)
    if not path.exists():
        typer.secho(f"Error: Agent directory '{path}' not found.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    if dry_run:
        typer.echo(f"Dry run: Would delete all agent data in directory '{path}'.")
        # To show more details, we might need to load parts of the agent config or store,
        # but for a simple dry run, just stating the directory is often enough.
        return

    if not force:
        if not typer.confirm(f"Are you sure you want to delete all agent data in '{path}'? This cannot be undone.", abort=True):
            return # typer.confirm with abort=True will exit if user says no.

    try:
        shutil.rmtree(path)
        typer.echo(f"Successfully cleared agent data at {path}")
    except Exception as e:
        typer.secho(f"Error clearing agent data at '{path}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


# --- Top-Level Commands ---
@app.command("ingest", help="Ingests text from a file or directory into the agent's memory.")
def ingest_command( # Renamed
    ctx: typer.Context,
    source: Path = typer.Argument(..., help="Path to the text file or directory.", exists=True, resolve_path=True),
    tau_override: Optional[float] = typer.Option(None, "--tau", "-t", help="Override agent's similarity threshold for this ingestion."),
    json_output: bool = typer.Option(False, "--json", help="Output ingestion summary in JSON format."),
    # Add relevant CLI overrides for VS/EF if they should affect ingestion behavior beyond agent's saved config
) -> None:
    agent_dir_str = ctx.obj.get("compact_memory_path")
    if not agent_dir_str: typer.secho("Error: Memory path not set.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    agent = _load_agent_cli(
        agent_dir_str,
        cli_vs_type=ctx.obj.get("cli_vector_store_type"), # Pass global CLI options
        cli_vs_path=ctx.obj.get("cli_vector_store_path"),
        cli_vs_collection=ctx.obj.get("cli_vector_store_collection"),
        cli_emb_provider=ctx.obj.get("cli_embedding_provider_type"),
        cli_emb_model=ctx.obj.get("cli_embedding_model_name"),
        cli_emb_device=ctx.obj.get("cli_embedding_device"),
        cli_emb_dim=ctx.obj.get("cli_embedding_dim"),
    )

    original_tau = agent.similarity_threshold
    if tau_override is not None:
        if not 0.5 <= tau_override <= 0.95:
            typer.secho("Error: --tau must be between 0.5 and 0.95.", err=True, fg=typer.colors.RED); raise typer.Exit(1)
        agent.similarity_threshold = tau_override
        logging.info(f"Overriding agent similarity threshold with {tau_override} for this ingestion.")

    # Simplified ingestion loop - run_experiment is more for batch evaluations
    # This is a direct ingestion using agent.add_memory
    files_to_ingest = []
    if source.is_file():
        files_to_ingest.append(source)
    elif source.is_dir():
        # Simple non-recursive text file find, can be enhanced with glob/pattern
        files_to_ingest.extend(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    if not files_to_ingest:
        typer.secho(f"No text files found to ingest at source: {source}", fg=typer.colors.YELLOW); return

    total_chunks_ingested = 0
    start_time = time.time()

    with PersistenceLock(Path(agent_dir_str)): # Lock during the whole ingestion batch
        for file_path in files_to_ingest:
            typer.echo(f"Ingesting from {file_path}...")
            try:
                text_content = file_path.read_text(encoding="utf-8")
                results = agent.add_memory(text_content, source_document_id=file_path.name)
                total_chunks_ingested += len(results)
            except Exception as e:
                typer.secho(f"Error ingesting {file_path}: {e}", fg=typer.colors.RED, err=True)

        # Persist changes after all files in this run are processed
        agent.save_agent(agent_dir_str)

    if tau_override is not None: # Restore original tau if it was overridden
        agent.similarity_threshold = original_tau
        # Note: agent instance is modified, but not re-saved here with original tau. This is fine for CLI.

    elapsed_time = time.time() - start_time
    metrics = {
        "files_processed": len(files_to_ingest),
        "total_chunks_ingested": total_chunks_ingested,
        "time_taken_seconds": round(elapsed_time, 2),
        **agent.prototype_system.metrics # Include metrics from PSS
    }
    if json_output:
        typer.echo(json.dumps(metrics))
    else:
        typer.echo("\nIngestion Summary:")
        for k, v in metrics.items():
            typer.echo(f"  {k}: {v}")
    typer.echo(f"Ingestion from '{source}' complete.")


@app.command("query", help='Queries the Compact Memory agent and returns an AI-generated response.')
def query_command( # Renamed
    ctx: typer.Context,
    query_text: str = typer.Argument(..., help="The query text to send to the agent."),
    show_prompt_tokens: bool = typer.Option(False, "--show-prompt-tokens", help="Display prompt token count."),
    # Add relevant CLI overrides for VS/EF if they should affect query behavior
) -> None:
    agent_dir_str = ctx.obj.get("compact_memory_path")
    if not agent_dir_str: typer.secho("Error: Memory path not set.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    # Use global CLI options passed via ctx.obj for overrides when loading agent
    agent = _load_agent_cli(
        agent_dir_str,
        cli_vs_type=ctx.obj.get("cli_vector_store_type"),
        cli_vs_path=ctx.obj.get("cli_vector_store_path"),
        cli_vs_collection=ctx.obj.get("cli_vector_store_collection"),
        cli_emb_provider=ctx.obj.get("cli_embedding_provider_type"),
        cli_emb_model=ctx.obj.get("cli_embedding_model_name"),
        cli_emb_device=ctx.obj.get("cli_embedding_device"),
        cli_emb_dim=ctx.obj.get("cli_embedding_dim"),
    )

    # LLM and compression strategy setup (remains similar)
    # ... (LLM setup as before) ...
    final_model_id = ctx.obj.get("default_model_id")
    final_strategy_id = ctx.obj.get("default_strategy_id")

    if final_model_id is None:
        typer.secho("Error: Default Model ID not specified.", fg=typer.colors.RED, err=True); raise typer.Exit(1)

    try:
        # Ensure agent._chat_model is initialized if not already (e.g. by load_agent or here)
        if not hasattr(agent, '_chat_model') or agent._chat_model is None:
            agent._chat_model = local_llm.LocalChatModel(model_name=final_model_id)
            agent._chat_model.load_model() # Explicitly load if needed
    except RuntimeError as exc:
        typer.secho(f"Error loading chat model '{final_model_id}': {exc}", err=True, fg=typer.colors.RED); raise typer.Exit(1)

    # Use agent's active_memory_manager
    mgr = agent.active_memory_manager
    comp_strategy_instance = None
    if final_strategy_id and final_strategy_id.lower() != "none":
        try:
            comp_cls = get_compression_strategy(final_strategy_id)
            comp_strategy_instance = comp_cls()
        except KeyError:
            typer.secho(f"Error: Unknown compression strategy '{final_strategy_id}'.", err=True, fg=typer.colors.RED); raise typer.Exit(1)

    # Lock the agent directory for the query operation if it involves any writes (e.g. AMM state changes)
    # For read-only query, lock might be optional or shared. PSS might update things.
    with PersistenceLock(Path(agent_dir_str)):
        result = agent.receive_channel_message(
            "cli", query_text, mgr, compression=comp_strategy_instance
        )

    reply = result.get("reply")
    if reply:
        typer.echo(reply)
    else:
        typer.secho("The agent did not return a reply.", fg=typer.colors.YELLOW)

    if show_prompt_tokens and result.get("prompt_tokens") is not None:
        typer.echo(f"Prompt tokens: {result['prompt_tokens']}")

# ... (rest of the CLI file: compress, dev commands, config commands, etc. remain largely unchanged for this subtask,
# unless they also load/interact with an Agent instance, in which case they'd need similar _load_agent_cli usage)

# Ensure the old load_agent from .utils is no longer used if it was previously.
# The new way is Agent.load_agent() called by _load_agent_cli helper.

# Final check: The old init command's InMemoryVectorStore instantiation and meta saving needs to be replaced
# by the new _create_dependencies_from_config and agent.save_agent() logic.
# The old validate_agent_storage used InMemoryVectorStore directly, now it should use _load_agent_cli.
# The old clear command also needs to be careful about what it's deleting if VS is external.
# For now, clear will remove the whole agent directory. More granular clearing would be an enhancement.

if __name__ == "__main__":
    app()
