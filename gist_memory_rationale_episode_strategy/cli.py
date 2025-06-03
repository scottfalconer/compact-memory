from __future__ import annotations

import typer
from pathlib import Path
from typing import Optional
from rich.table import Table
from rich.console import Console

from .storage import EpisodeStorage

console = Console()

episodes_app = typer.Typer(help="Inspect rationale-enhanced episode memory.")


@episodes_app.command("list")
def episodes_list(
    store_path: Path = typer.Argument(..., help="Episode storage directory")
) -> None:
    """List available episodes"""
    store = EpisodeStorage(store_path)
    table = Table("id", "summary", "tags")
    for ep in store.episodes:
        table.add_row(ep.id, ep.summary_gist[:40], ",".join(ep.tags))
    console.print(table)


@episodes_app.command("show")
def episodes_show(
    episode_id: str,
    store_path: Path = typer.Argument(..., help="Episode storage directory"),
) -> None:
    """Show details for an episode"""
    store = EpisodeStorage(store_path)
    for ep in store.episodes:
        if ep.id == episode_id:
            console.print(ep.summary_gist)
            console.print(f"tags: {', '.join(ep.tags)}")
            console.print(f"decisions: {len(ep.decisions)}")
            return
    typer.secho("Episode not found", err=True, fg=typer.colors.RED)


@episodes_app.command("tag")
def episodes_tag(
    episode_id: str,
    add: Optional[str] = typer.Option(None, "--add", help="Tag to add"),
    store_path: Path = typer.Argument(..., help="Episode storage directory"),
) -> None:
    """Add a tag to an episode"""
    store = EpisodeStorage(store_path)
    for ep in store.episodes:
        if ep.id == episode_id:
            if add and add not in ep.tags:
                ep.tags.append(add)
                store.update_episode(ep)
            console.print(f"tags: {', '.join(ep.tags)}")
            return
    typer.secho("Episode not found", err=True, fg=typer.colors.RED)
