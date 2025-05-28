import sys


def main(argv=None) -> None:
    """Entry point for the ``gist-memory`` command.

    When invoked without arguments, launch the Textual TUI. Otherwise fall back
    to the CLI implementation.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 0:
        from .tui import run_tui

        run_tui()
    else:
        from .cli import app

        app()


if __name__ == "__main__":
    main()
