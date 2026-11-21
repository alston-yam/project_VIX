"""Configure logging to both the console and ``logs/pipeline.log``."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path, level: int = logging.INFO, filename: str = "pipeline.log") -> Path:
    """Attach a console handler and a file handler to the root logger.

    Calling this twice does not duplicate handlers. Returns the log file path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    # Quieten noisy third-party loggers.
    for noisy in ("yfinance", "urllib3", "matplotlib", "peewee", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a short name (``vrp.x`` -> ``x``)."""
    return logging.getLogger(name.replace("vrp.", ""))
