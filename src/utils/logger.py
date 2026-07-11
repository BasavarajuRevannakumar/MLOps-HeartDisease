"""
Centralized logging configuration module.
"""

import logging
from pathlib import Path

from src.utils.config import LOG_DIR


LOG_FILE = LOG_DIR / "heart_mlops.log"


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__)

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
