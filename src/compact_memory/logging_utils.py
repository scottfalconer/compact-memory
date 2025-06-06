import logging
from pathlib import Path


def configure_logging(log_file: Path, level: int = logging.INFO) -> None:
    """Configure Python logging to write to ``log_file``."""
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_file))
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


__all__ = ["configure_logging"]
