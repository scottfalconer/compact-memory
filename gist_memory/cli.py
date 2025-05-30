import json
import shutil
from pathlib import Path
from typing import Optional
import logging

import typer
import portalocker
from rich.table import Table
from rich.console import Console

from .logging_utils import configure_logging



from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .chunker import SentenceWindowChunker
from .embedding_pipeline import embed_text, EmbeddingDimensionMismatchError
from .utils import load_agent
from .config import DEFAULT_BRAIN_PATH

app = typer.Typer(help="Gist Memory command line interface")
console = Console()
VERBOSE = False


@app.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Write debug logs to this file"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Configure logging before executing commands."""
    if log_file:
        level = logging.DEBUG if verbose else logging.INFO
        configure_logging(log_file, level)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG)
    global VERBOSE
    VERBOSE = verbose
    ctx.obj = {"verbose": verbose}


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
        f"Try running gist-memory validate {path} for more details or restore from a backup.",
        err=True,
    )
    raise typer.Exit(code=1)


def _load_agent(path: Path) -> Agent:
    try:
        return load_agent(path)
    except Exception as exc:
        _corrupt_exit(path, exc)


@app.command()
def init(
    directory: str = typer.Argument(DEFAULT_BRAIN_PATH),
    *,
    agent_name: str = "default",
    model_name: str = "all-MiniLM-L6-v2",
    tau: float = 0.8,
    alpha: float = 0.1,
    chunker: str = "sentence_window",
) -> None:
    """Create a new agent store."""
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        typer.secho(
            "Directory already exists and is not empty",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if not 0.5 <= tau <= 0.95:
        typer.secho(
            "Error: --tau must be between 0.5 and 0.95.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    try:
        dim = int(embed_text(["dim"]).shape[1])
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    store = JsonNpyVectorStore(
        path=str(path), embedding_model=model_name, embedding_dim=dim
    )
    store.meta.update(
        {
            "agent_name": agent_name,
            "tau": tau,
            "alpha": alpha,
            "chunker": chunker,
        }
    )
    store.save()
    typer.echo(f"Initialized agent at {directory}")


@app.command()
def add(
    *,
    agent_name: str = typer.Option(
        DEFAULT_BRAIN_PATH, help="Path to the agent directory"
    ),
    text: Optional[str] = typer.Option(None, help="Text to add"),
    file: Optional[Path] = typer.Option(None, help="Text file to add"),
    source_id: Optional[str] = typer.Option(None, help="Source id"),
    actor: Optional[str] = typer.Option(None, help="Actor"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not modify the store"),
) -> None:
    """Ingest new text into the agent."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        file_text = file.read_text() if file else ""
    except Exception as exc:
        typer.secho(
            f"Error: Could not read file '{file}'. Reason: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    input_text = text or file_text
    if not input_text:
        typer.secho("No text provided", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with PersistenceLock(path):
        agent = _load_agent(path)
        try:
            results = agent.add_memory(input_text, who=actor)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
        agent.store.save()
        chunks = agent.chunker.chunk(input_text)
        from tqdm import tqdm

        bar = tqdm(total=len(chunks), desc="Adding", disable=False)

        def cb(i, total, spawned, pid, sim):
            bar.update(1)
            if VERBOSE:
                act = "spawned" if spawned else "updated"
                sim_str = f" (similarity: {sim:.2f})" if sim is not None else ""
                typer.echo(f"Chunk {i}/{total}: {act} {pid}{sim_str}")

        results = agent.add_memory(
            input_text, who=actor, progress_callback=cb, save=not dry_run
        )
        bar.close()
        if not dry_run:
            agent.store.save()
    if dry_run:
        spawned = sum(1 for r in results if r.get("spawned"))
        updated = sum(1 for r in results if not r.get("spawned"))
        typer.echo(
            f"Would spawn {spawned} new prototype{'s' if spawned!=1 else ''} and update {updated} existing ones."
        )
        return
    proto_map = {p.prototype_id: p for p in agent.store.prototypes}
    for r in results:
        if r.get("duplicate"):
            typer.echo("Duplicate text skipped")
            continue
        action = "Spawned new prototype" if r.get("spawned") else "Updated prototype"
        sim = r.get("sim")
        proto = proto_map.get(r["prototype_id"])
        summary = proto.summary_text if proto else ""
        sim_str = f" (similarity: {sim:.2f})" if sim is not None else ""
        typer.echo(
            f"Successfully added memory. {action} {r['prototype_id']}{sim_str}. Summary: '{summary}'"
        )


@app.command()
def query(
    *,
    agent_name: str = typer.Option(DEFAULT_BRAIN_PATH, help="Agent directory"),
    query_text: str = typer.Option(..., help="Query text"),
    k_prototypes: int = typer.Option(1, help="Number of prototypes"),
    k_memories: int = typer.Option(3, help="Number of memories"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Query stored beliefs."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    with PersistenceLock(path):
        agent = _load_agent(path)
        try:
            res = agent.query(
                query_text, top_k_prototypes=k_prototypes, top_k_memories=k_memories
            )
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
    if json_output:
        typer.echo(json.dumps(res))
        return
    for proto in res["prototypes"]:
        console.print(
            f"[bold]{proto['id']}[/bold] {proto['summary']} ({proto['sim']:.2f})"
        )
    for mem in res["memories"]:
        console.print(f"  {mem['text']} ({mem['sim']:.2f})")


@app.command("list-beliefs")
def list_beliefs(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
    sort: str = typer.Option("", help="Sort order"),
) -> None:
    """List all belief prototypes."""
    path = Path(agent_name)
    agent = _load_agent(path)
    protos = agent.store.prototypes
    if sort == "strength":
        protos = sorted(protos, key=lambda p: p.strength, reverse=True)
    table = Table("id", "strength", "confidence", "summary", title="Beliefs")
    for p in protos:
        table.add_row(
            p.prototype_id,
            f"{p.strength:.2f}",
            f"{p.confidence:.2f}",
            p.summary_text[:60],
        )
    console.print(table)


@app.command()
def stats(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show statistics about the store."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        store = JsonNpyVectorStore(path=str(path))
    except Exception as exc:
        _corrupt_exit(path, exc)
    size = shutil.disk_usage(path).used
    data = {
        "prototypes": len(store.prototypes),
        "active_memories": len(store.memories),
        "archived_memories": 0,
        "disk_size": size,
        "last_decay": store.meta.get("last_decay_ts"),
    }
    if json_output:
        typer.echo(json.dumps(data))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


@app.command()
def talk(
    *,
    agent_name: str = typer.Option(DEFAULT_BRAIN_PATH, help="Agent directory"),
    message: str = typer.Option(..., help="Message to send"),
    model_name: str = typer.Option("distilgpt2", help="Local chat model"),
) -> None:
    """Talk to the brain using the same pathway as chat sessions."""

    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    with PersistenceLock(path):
        agent = _load_agent(path)

        from .local_llm import LocalChatModel

        try:
            agent._chat_model = LocalChatModel(model_name=model_name)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)

        from .active_memory_manager import ActiveMemoryManager

        mgr = ActiveMemoryManager()
        result = agent.receive_channel_message("cli", message, mgr)
        reply = result.get("reply")
        if reply:
            typer.echo(reply)

    return


@app.command()
def validate(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
) -> None:
    """Validate the store metadata and embeddings."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        JsonNpyVectorStore(path=str(path))
    except EmbeddingDimensionMismatchError as exc:
        typer.secho(
            f"Embedding dimension mismatch: {exc}", err=True, fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover - unexpected errors
        typer.secho(f"Error loading agent: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo("Store is valid")


@app.command()
def clear(
    agent_name: str = typer.Argument(DEFAULT_BRAIN_PATH, help="Agent directory"),
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not delete files"),
) -> None:
    """Delete all data in the store."""
    path = Path(agent_name)
    if not path.exists():
        typer.secho(
            f"Error: Agent directory '{path}' not found or is invalid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if dry_run:
        store = JsonNpyVectorStore(path=str(path))
        typer.echo(
            f"Would delete {len(store.prototypes)} prototypes and {len(store.memories)} memories."
        )
        return
    if not yes:
        if not typer.confirm(
            f"Delete {path}?", abort=True
        ):  # pragma: no cover - user abort
            return
    if path.exists():
        shutil.rmtree(path)
        typer.echo(f"Deleted {path}")
    else:
        typer.secho("Directory not found", err=True, fg=typer.colors.RED)


@app.command("download-model")
def download_model(
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2", help="SentenceTransformer model name"
    )
) -> None:
    """Pre-download a local embedding model."""
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    bar = tqdm(total=1, desc="Downloading model", disable=False)
    SentenceTransformer(model_name)
    bar.update(1)
    bar.close()
    typer.echo(f"Downloaded {model_name}")


@app.command("download-chat-model")
def download_chat_model(
    model_name: str = typer.Option("distilgpt2", help="Local causal LM name")
) -> None:
    """Pre-download a local chat model for ``talk`` mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    bar = tqdm(total=1, desc="Downloading chat model", disable=False)
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    bar.update(1)
    bar.close()
    typer.echo(f"Downloaded {model_name}")


@app.command("experiment")
def run_experiment_cmd(
    dataset: Path = typer.Argument(..., help="Text file to ingest"),
    work_dir: Optional[Path] = typer.Option(
        None, help="Directory for the temporary store"
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--tau", help="Similarity threshold"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run a simple ingestion experiment."""
    from .experiment_runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        dataset=dataset, similarity_threshold=similarity_threshold, work_dir=work_dir
    )
    metrics = run_experiment(cfg)
    if json_output:
        typer.echo(json.dumps(metrics))
    else:
        for k, v in metrics.items():
            typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
