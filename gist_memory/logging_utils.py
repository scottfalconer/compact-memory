import logging
from pathlib import Path


def get_library_logger() -> logging.Logger:
    """Return the root logger for the ``gist_memory`` package."""
    return logging.getLogger("gist_memory")


def set_library_log_level(level: int, *, add_basic_handler: bool = False) -> None:
    """Set the log level for the ``gist_memory`` logger.

    Parameters
    ----------
    level:
        Logging verbosity level.
    add_basic_handler:
        When ``True`` and the root logger has no handlers, ``logging.basicConfig``
        will be called so log messages appear on ``stderr``.  This is handy in
        simple scripts or notebooks.
    """

    logger = get_library_logger()
    logger.setLevel(level)

    if add_basic_handler and not logging.getLogger().handlers:
        logging.basicConfig(level=level)


def configure_logging(log_file: Path, level: int = logging.INFO) -> None:
    """Configure Python logging to write to ``log_file``."""
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_file))
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = get_library_logger()
    logger.setLevel(level)
    logger.addHandler(handler)


__all__ = ["configure_logging", "get_library_logger", "set_library_log_level"]
