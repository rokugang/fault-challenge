from __future__ import annotations

import logging
from pathlib import Path

from . import config

LOG_PATH: Path = config.LOGS_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return module logger with project defaults."""
    return logging.getLogger(name)
