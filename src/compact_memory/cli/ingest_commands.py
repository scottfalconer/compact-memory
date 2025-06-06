import typer

# ingest_app = typer.Typer() # Not needed for a single, simple command

def ingest_command() -> None: # Renamed from ingest
    typer.secho("Ingestion functionality was removed from the CLI in this version.", fg=typer.colors.YELLOW)
    typer.echo("Please refer to documentation for current methods of adding data to memory stores, such as using the 'compress --memory-path ...' command.")
    raise typer.Exit(code=1) # Exit with error to indicate it's not a valid command for use

# This file will be very simple as the command is deprecated.
# No complex imports needed.
# The command will be imported into main.py and added to the app.
# It's good practice to still make it a function for consistency.
