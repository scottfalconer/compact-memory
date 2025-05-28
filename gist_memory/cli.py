import click
from pathlib import Path
import sys

from .memory_creation import (
    IdentityMemoryCreator,
    ExtractiveSummaryCreator,
    ChunkMemoryCreator,
    LLMSummaryCreator,
    AgenticMemoryCreator,
)
from .store import (
    PrototypeStore,
    JSONVectorStore,
    ChromaVectorStore,
    CloudVectorStore,
)
from tqdm import tqdm
from .embedder import get_embedder, LocalEmbedder


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
    type=click.Choice(["identity", "extractive", "chunk", "llm", "agentic"]),
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
@click.option(
    "--min-threshold",
    default=0.05,
    type=float,
    show_default=True,
    help="Minimum adaptive threshold",
)
@click.option(
    "--decay-exponent",
    default=0.5,
    type=float,
    show_default=True,
    help="Exponent controlling threshold decay",
)
@click.option(
    "--vector-store",
    type=click.Choice(["json", "chroma", "cloud"]),
    default="json",
    show_default=True,
    help="Backend storage implementation",
)
@click.option(
    "--db-path",
    default="gist_memory_db",
    show_default=True,
    help="Path for persistent storage",
)
@click.pass_context
def cli(
    ctx,
    embedder,
    model_name,
    memory_creator,
    threshold,
    min_threshold,
    decay_exponent,
    vector_store,
    db_path,
):
    """Gist Memory Agent CLI."""
    ctx.obj = {
        "embedder": get_embedder(embedder, model_name),
        "memory_creator": {
            "identity": IdentityMemoryCreator,
            "extractive": ExtractiveSummaryCreator,
            "chunk": ChunkMemoryCreator,
            "llm": LLMSummaryCreator,
            "agentic": AgenticMemoryCreator,
        }[memory_creator](),
        "threshold": threshold,
        "min_threshold": min_threshold,
        "decay_exponent": decay_exponent,
        "store_cls": {
            "json": JSONVectorStore,
            "chroma": ChromaVectorStore,
            "cloud": CloudVectorStore,
        }[vector_store],
        "db_path": db_path,
    }


@cli.command()
@click.argument("source", nargs=-1)
@click.pass_obj
def ingest(obj, source):
    """Ingest text or contents of a file/directory."""
    creator = obj["memory_creator"]
    store = obj["store_cls"](
        embedder=obj["embedder"],
        threshold=obj["threshold"],
        min_threshold=obj["min_threshold"],
        decay_exponent=obj["decay_exponent"],
        path=obj["db_path"],
    )

    def process_chunks(chunks: list[str]) -> None:
        with tqdm(chunks, desc="Ingesting", unit="mem", disable=not sys.stderr.isatty()) as bar:
            for chunk in bar:
                before = store.prototype_count()
                mem = store.add_memory(chunk)
                after = store.prototype_count()
                action = "Created" if after > before else "Updated"
                tqdm.write(f"{action} prototype {mem.prototype_id} with memory {mem.id}")

    if len(source) == 1:
        path = Path(source[0])
        if path.exists():
            texts: list[str] = []
            if path.is_file():
                texts.append(path.read_text())
            elif path.is_dir():
                for f in sorted(path.glob("*.txt")):
                    texts.append(f.read_text())
            chunks: list[str] = []
            for text in texts:
                chunks.extend(creator.create_all(text))
            process_chunks(chunks)
            return

    content = " ".join(source)
    chunks = creator.create_all(content)
    process_chunks(chunks)


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--top", default=3, help="Number of results")
@click.pass_obj
def query(obj, text, top):
    """Query the store."""
    content = " ".join(text)
    store = obj["store_cls"](
        embedder=obj["embedder"],
        threshold=obj["threshold"],
        min_threshold=obj["min_threshold"],
        decay_exponent=obj["decay_exponent"],
        path=obj["db_path"],
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
    store = obj["store_cls"](
        embedder=obj["embedder"],
        threshold=obj["threshold"],
        min_threshold=obj["min_threshold"],
        decay_exponent=obj["decay_exponent"],
        path=obj["db_path"],
    )
    memories = store.decode_prototype(prototype_id, n=top)
    if not memories:
        click.echo("Prototype not found")
        return
    for mem in memories:
        click.echo(f"{mem.id}: {mem.text}")


@cli.command(name="summarize")
@click.argument("prototype_id")
@click.option("--max-words", default=50, help="Length of summary")
@click.pass_obj
def summarize_prototype(obj, prototype_id, max_words):
    """Show a simple summary for a prototype."""
    store = obj["store_cls"](
        embedder=obj["embedder"],
        threshold=obj["threshold"],
        min_threshold=obj["min_threshold"],
        decay_exponent=obj["decay_exponent"],
        path=obj["db_path"],
    )
    summary = store.summarize_prototype(prototype_id, max_words=max_words)
    if summary is None:
        click.echo("Prototype not found")
    else:
        click.echo(summary)


@cli.command(name="dump")
@click.option("--prototype-id", default=None, help="Only dump memories for this prototype")
@click.pass_obj
def dump_memories(obj, prototype_id):
    """Dump all memories, optionally for a given prototype."""
    store = obj["store_cls"](
        embedder=obj["embedder"],
        threshold=obj["threshold"],
        min_threshold=obj["min_threshold"],
        decay_exponent=obj["decay_exponent"],
        path=obj["db_path"],
    )
    memories = store.dump_memories(prototype_id=prototype_id)
    for mem in memories:
        click.echo(f"[{mem.prototype_id}] {mem.text}")


@cli.command(name="download-model")
@click.option(
    "--model-name",
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="Embedding model to pre-download",
)
def download_model(model_name: str) -> None:
    """Pre-fetch a local embedding model."""
    try:
        LocalEmbedder(model_name=model_name, local_files_only=False)
    except Exception as exc:  # pragma: no cover - passthrough
        raise click.ClickException(str(exc))
    click.echo(f"Downloaded model '{model_name}'")


if __name__ == "__main__":
    cli()
