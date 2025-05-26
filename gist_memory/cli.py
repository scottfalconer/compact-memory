import click
from pathlib import Path

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
@click.argument("source", nargs=-1)
@click.pass_obj
def ingest(obj, source):
    """Ingest text or contents of a file/directory."""
    creator = obj["memory_creator"]
    store = PrototypeStore(
        embedder=obj["embedder"], threshold=obj["threshold"]
    )

    if len(source) == 1:
        path = Path(source[0])
        if path.exists():
            texts: list[str] = []
            if path.is_file():
                texts.append(path.read_text())
            elif path.is_dir():
                for f in sorted(path.glob("*.txt")):
                    texts.append(f.read_text())
            for text in texts:
                mem = store.add_memory(creator.create(text))
                click.echo(
                    f"Stored memory {mem.id} in prototype {mem.prototype_id}"
                )
            return

    content = " ".join(source)
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


@cli.command(name="decode")
@click.argument("prototype_id")
@click.option("--top", default=1, help="Number of memories to show")
@click.pass_obj
def decode_prototype(obj, prototype_id, top):
    """Show example memories for a prototype."""
    store = PrototypeStore(
        embedder=obj["embedder"], threshold=obj["threshold"]
    )
    memories = store.decode_prototype(prototype_id, n=top)
    if not memories:
        click.echo("Prototype not found")
        return
    for mem in memories:
        click.echo(f"{mem.id}: {mem.text}")


@cli.command(name="dump")
@click.option("--prototype-id", default=None, help="Only dump memories for this prototype")
@click.pass_obj
def dump_memories(obj, prototype_id):
    """Dump all memories, optionally for a given prototype."""
    store = PrototypeStore(
        embedder=obj["embedder"], threshold=obj["threshold"]
    )
    memories = store.dump_memories(prototype_id=prototype_id)
    for mem in memories:
        click.echo(f"[{mem.prototype_id}] {mem.text}")


if __name__ == "__main__":
    cli()
