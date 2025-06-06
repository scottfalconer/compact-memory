from .main import app

# This makes the `app` object available when the `compact_memory.cli` package is imported,
# e.g., for potential testing or programmatic invocation setups.

# Example usage (hypothetical, not part of the CLI itself):
# from compact_memory.cli import app
# from typer.testing import CliRunner
# runner = CliRunner()
# result = runner.invoke(app, ["engine", "list"])
# print(result.stdout)

# Ensure all submodules that define commands (engine_commands, config_commands, etc.)
# are correctly structured and that their Typer apps or command functions are imported
# into cli.main.py where they are added to the main 'app'.

# This __init__.py primarily serves to define the public API of the 'cli' package.
# By exporting 'app', we designate it as the primary entry point or interface for this package.
# No other changes seem necessary here based on the refactoring.
# All command definitions and Typer app instantiations are within their respective modules
# and then aggregated in `main.py`.
# Corrected imports in main.py and sub-command files are crucial for this to work.
# e.g. `from compact_memory.config import ...` instead of `from ..config import ...`
# and `from .engine_commands import engine_app` within main.py.
# The `main_callback` in `main.py` correctly initializes `ctx.obj` and handles global options.
# Plugin loading (`load_plugins`, `enable_all_experimental_engines`) is now in `main_callback`.
# The logic for prompting for `memory_path` if not set has been kept in `main_callback`.
# All looks good for this __init__.py.
