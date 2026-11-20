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


class _CurrentStderrLogger(structlog.PrintLogger):
    """A PrintLogger that always writes to the current ``sys.stderr``.

    Test runners and servers replace ``sys.stderr``; binding the stream once at
    configuration time would make later log calls fail on a closed file.
    """

    def __init__(self) -> None:
        super().__init__(file=sys.stderr)

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)


class _CurrentStderrFactory:
    """Logger factory used by :func:`configure_logging`."""

    def __call__(self, *args: object) -> _CurrentStderrLogger:
        del args
        return _CurrentStderrLogger()


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
        logger_factory=_CurrentStderrFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger; call :func:`configure_logging` first."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
