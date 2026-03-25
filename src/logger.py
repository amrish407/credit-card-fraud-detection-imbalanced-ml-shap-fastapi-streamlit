import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src import config


def _build_log_path() -> Path:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    return config.LOG_DIR / "project.log"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(_build_log_path(), maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
