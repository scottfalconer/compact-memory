import click

from .memory_creation import IdentityMemoryCreator
from .store import PrototypeStore


@click.group()
@click.option("--db-dir", default="gist_memory_db", help="Directory for database")
@click.pass_context
def cli(ctx, db_dir: str):
    """Gist Memory Agent CLI."""
    ctx.obj = {"db_dir": db_dir}


@cli.command()
@click.argument("text", nargs=-1)
@click.pass_context
def ingest(ctx, text):
    """Ingest a memory from TEXT."""
    content = " ".join(text)
    creator = IdentityMemoryCreator()
    store = PrototypeStore(path=ctx.obj["db_dir"])
    mem = store.add_memory(creator.create(content))
    click.echo(f"Stored memory {mem.id} in prototype {mem.prototype_id}")


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--top", default=3, help="Number of results")
@click.pass_context
def query(ctx, text, top):
    """Query the store."""
    content = " ".join(text)
    store = PrototypeStore(path=ctx.obj["db_dir"])
    results = store.query(content, n=top)
    for mem in results:
        click.echo(f"[{mem.prototype_id}] {mem.text}")


if __name__ == "__main__":
    cli()
