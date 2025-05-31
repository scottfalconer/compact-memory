from gist_memory.tui import run_tui
from gist_memory.config import DEFAULT_BRAIN_PATH


def main(path: str = DEFAULT_BRAIN_PATH) -> None:
    """Entry point for ``gist-run`` launching the unified TUI."""
    run_tui(path)


if __name__ == "__main__":
    main()
