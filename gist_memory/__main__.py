import sys
import logging
from pathlib import Path

try:  # optional pretty tracebacks
    from rich.traceback import install as install_rich_traceback

    install_rich_traceback()
except Exception:  # pragma: no cover - rich may not be installed
    pass

from .logging_utils import configure_logging


def main(argv=None) -> None:
    """Entry point for the ``gist-memory`` command.

    When invoked without arguments, launch the Textual TUI. Otherwise fall back
    to the CLI implementation.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    log_file = None
    verbose = False
    if "--log-file" in args:
        idx = args.index("--log-file")
        log_file = args[idx + 1]
        del args[idx: idx + 2]
    if "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    if log_file:
        level = logging.DEBUG if verbose else logging.INFO
        configure_logging(Path(log_file), level)

    sys.argv = [sys.argv[0]] + args

    if len(args) == 0:
        from .tui import run_tui

        run_tui()
    else:
        from .cli import app

        app()


if __name__ == "__main__":
    main()
