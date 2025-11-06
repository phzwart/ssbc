"""Logging utilities for SSBC package.

This module provides centralized logging configuration and utilities.
"""

import logging
import sys
from typing import Any

# Configure package logger
_logger = logging.getLogger("ssbc")


def get_logger(name: str | None = None) -> logging.Logger:
    """Get logger instance for a module.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns the package logger.
        If provided, returns a child logger of the package logger.

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> from ssbc._logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.debug("Debug message")
    >>> logger.info("Info message")
    """
    if name is None:
        return _logger
    # Return child logger of package logger
    return _logger.getChild(name.replace("ssbc.", ""))


def configure_logging(level: int = logging.WARNING, stream: Any = sys.stderr) -> None:
    """Configure logging for the SSBC package.

    Parameters
    ----------
    level : int, default=logging.WARNING
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    stream : file-like object, default=sys.stderr
        Stream for log output

    Examples
    --------
    >>> from ssbc._logging import configure_logging
    >>> import logging
    >>> configure_logging(level=logging.INFO)
    """
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.propagate = False


# Initialize with default WARNING level on import
configure_logging()
