import sys
import logging
from pathlib import Path

try:  # optional pretty tracebacks
    from rich.traceback import install as install_rich_traceback

    install_rich_traceback()
except Exception:  # pragma: no cover - rich may not be installed
    pass

from .logging_utils import configure_logging, set_library_log_level


def main(argv=None) -> None:
    """Entry point for the ``gist-memory`` command.

    This module used to launch a Textual TUI when ``gist-memory`` was executed
    without any arguments. The TUI has been removed so we now always invoke the
    Typer based CLI regardless of the arguments passed.
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
    elif verbose:
        set_library_log_level(logging.DEBUG, add_basic_handler=True)

    sys.argv = [sys.argv[0]] + args

    from .cli import app

    app()


if __name__ == "__main__":
    main()
