import click

from .memory_creation import IdentityMemoryCreator
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
@click.pass_context
def cli(ctx, embedder, model_name):
    """Gist Memory Agent CLI."""
    ctx.obj = {
        "embedder": get_embedder(embedder, model_name),
    }


@cli.command()
@click.argument("text", nargs=-1)
@click.pass_obj
def ingest(obj, text):
    """Ingest a memory from TEXT."""
    content = " ".join(text)
    creator = IdentityMemoryCreator()
    store = PrototypeStore(embedder=obj["embedder"])
    mem = store.add_memory(creator.create(content))
    click.echo(f"Stored memory {mem.id} in prototype {mem.prototype_id}")


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--top", default=3, help="Number of results")
@click.pass_obj
def query(obj, text, top):
    """Query the store."""
    content = " ".join(text)
    store = PrototypeStore(embedder=obj["embedder"])
    results = store.query(content, n=top)
    for mem in results:
        click.echo(f"[{mem.prototype_id}] {mem.text}")


if __name__ == "__main__":
    cli()
