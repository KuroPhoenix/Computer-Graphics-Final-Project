import logging
import sys
from pathlib import Path
from typing import Optional

from configs import config

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _resolve_level(level_name: Optional[str]) -> int:
    if not level_name:
        return logging.INFO
    return getattr(logging, level_name.upper(), logging.INFO)


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: Optional[bool] = None,
) -> logging.Logger:
    """
    Configure root logger once with optional console and file output.
    Subsequent calls are no-ops if handlers already exist.
    """
    root = logging.getLogger()
    if root.handlers:
        return root

    resolved_level = _resolve_level(level or config.LOG_LEVEL)
    root.setLevel(resolved_level)

    formatter = logging.Formatter(LOG_FORMAT)

    use_console = config.LOG_CONSOLE if console is None else console
    if use_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    target_file = log_file if log_file is not None else config.LOG_FILE
    if target_file:
        path = Path(target_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring root logging on first use."""
    configure_logging()
    return logging.getLogger(name)
