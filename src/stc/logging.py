"""Structured logging configured once by the CLI composition root."""

from __future__ import annotations

import logging
import sys
from typing import Literal, cast

import structlog

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(
    level: str = "INFO",
    log_format: Literal["console", "json"] = "console",
) -> None:
    """Configure structlog for human-readable console or one-line JSON output."""
    numeric_level = _LOG_LEVELS.get(level.upper(), logging.INFO)
    renderer: structlog.types.Processor
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger; call :func:`configure_logging` first."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
