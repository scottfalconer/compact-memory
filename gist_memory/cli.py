import json
import shutil
from pathlib import Path
from typing import Optional

import typer
import portalocker
from rich.table import Table
from rich.console import Console

from .agent import Agent
from .json_npy_store import JsonNpyVectorStore
from .chunker import SentenceWindowChunker, _CHUNKER_REGISTRY
from .embedding_pipeline import embed_text

app = typer.Typer(help="Gist Memory command line interface")
console = Console()


class PersistenceLock:
    def __init__(self, path: Path) -> None:
        self.file = (path / ".lock").open("a+")

    def __enter__(self):
        portalocker.lock(self.file, portalocker.LockFlags.EXCLUSIVE)
        return self

    def __exit__(self, exc_type, exc, tb):
        portalocker.unlock(self.file)
        self.file.close()


def _load_agent(path: Path) -> Agent:
    store = JsonNpyVectorStore(path=str(path))
    chunker_id = store.meta.get("chunker", "sentence_window")
    chunker_cls = _CHUNKER_REGISTRY.get(chunker_id, SentenceWindowChunker)
    tau = float(store.meta.get("tau", 0.8))
    return Agent(store, chunker=chunker_cls(), similarity_threshold=tau)


@app.command()
def init(
    directory: str,
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
        typer.echo("Directory already exists and is not empty", err=True)
        raise typer.Exit(code=1)
    dim = int(embed_text(["dim"]).shape[1])
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
    agent_name: str = typer.Option(..., help="Path to the agent directory"),
    text: Optional[str] = typer.Option(None, help="Text to add"),
    file: Optional[Path] = typer.Option(None, help="Text file to add"),
    source_id: Optional[str] = typer.Option(None, help="Source id"),
    actor: Optional[str] = typer.Option(None, help="Actor"),
) -> None:
    """Ingest new text into the agent."""
    path = Path(agent_name)
    if not path.exists():
        typer.echo("Agent not found", err=True)
        raise typer.Exit(code=1)
    input_text = text or (file.read_text() if file else "")
    if not input_text:
        typer.echo("No text provided", err=True)
        raise typer.Exit(code=1)
    with PersistenceLock(path):
        agent = _load_agent(path)
        results = agent.add_memory(input_text)
        agent.store.save()
    for r in results:
        action = "spawned" if r.get("spawned") else "updated"
        sim = r.get("sim")
        typer.echo(f"{action} {r['prototype_id']} sim={sim:.2f}" if sim else action)


@app.command()
def query(
    *,
    agent_name: str = typer.Option(..., help="Agent directory"),
    query_text: str = typer.Option(..., help="Query text"),
    k_prototypes: int = typer.Option(1, help="Number of prototypes"),
    k_memories: int = typer.Option(3, help="Number of memories"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Query stored beliefs."""
    path = Path(agent_name)
    with PersistenceLock(path):
        agent = _load_agent(path)
        res = agent.query(
            query_text, top_k_prototypes=k_prototypes, top_k_memories=k_memories
        )
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
    agent_name: str = typer.Argument(..., help="Agent directory"),
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
    agent_name: str = typer.Argument(..., help="Agent directory"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show statistics about the store."""
    path = Path(agent_name)
    store = JsonNpyVectorStore(path=str(path))
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


@app.command("download-model")
def download_model(
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2", help="SentenceTransformer model name"
    )
) -> None:
    """Pre-download a local embedding model."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name)
    typer.echo(f"Downloaded {model_name}")


if __name__ == "__main__":
    app()
