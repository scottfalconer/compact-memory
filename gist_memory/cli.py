import click

from .memory_creation import IdentityMemoryCreator
from .store import PrototypeStore


@click.group()
def cli():
    """Gist Memory Agent CLI."""


@cli.command()
@click.argument("text", nargs=-1)
def ingest(text):
    """Ingest a memory from TEXT."""
    content = " ".join(text)
    creator = IdentityMemoryCreator()
    store = PrototypeStore()
    mem = store.add_memory(creator.create(content))
    click.echo(f"Stored memory {mem.id} in prototype {mem.prototype_id}")


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--top", default=3, help="Number of results")
def query(text, top):
    """Query the store."""
    content = " ".join(text)
    store = PrototypeStore()
    results = store.query(content, n=top)
    for mem in results:
        click.echo(f"[{mem.prototype_id}] {mem.text}")


if __name__ == "__main__":
    cli()
