import click

from .memory_creation import (
    IdentityMemoryCreator,
    ExtractiveSummaryCreator,
)
from .store import PrototypeStore
from .embedder import get_embedder


@click.group()
@click.option(
    "--embedder",
    type=click.Choice(["random", "openai", "local"]),
    default="random",
    help="Embedding backend",
)
@click.option("--model-name", default=None, help="Model name for the embedder")
@click.option(
    "--memory-creator",
    type=click.Choice(["identity", "extractive"]),
    default="identity",
    help="Memory creation strategy",
)
@click.option(
    "--threshold",
    default=0.4,
    type=float,
    show_default=True,
    help="Prototype assignment threshold",
)
@click.pass_context
def cli(ctx, embedder, model_name, memory_creator, threshold):
    """Gist Memory Agent CLI."""
    ctx.obj = {
        "embedder": get_embedder(embedder, model_name),
        "memory_creator": {
            "identity": IdentityMemoryCreator,
            "extractive": ExtractiveSummaryCreator,
        }[memory_creator](),
        "threshold": threshold,
    }


@cli.command()
@click.argument("text", nargs=-1)
@click.pass_obj
def ingest(obj, text):
    """Ingest a memory from TEXT."""
    content = " ".join(text)
    creator = obj["memory_creator"]
    store = PrototypeStore(
        embedder=obj["embedder"], threshold=obj["threshold"]
    )
    mem = store.add_memory(creator.create(content))
    click.echo(f"Stored memory {mem.id} in prototype {mem.prototype_id}")


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--top", default=3, help="Number of results")
@click.pass_obj
def query(obj, text, top):
    """Query the store."""
    content = " ".join(text)
    store = PrototypeStore(
        embedder=obj["embedder"], threshold=obj["threshold"]
    )
    results = store.query(content, n=top)
    for mem in results:
        click.echo(f"[{mem.prototype_id}] {mem.text}")


if __name__ == "__main__":
    cli()
