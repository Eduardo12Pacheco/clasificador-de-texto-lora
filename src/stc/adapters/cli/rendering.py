"""Shared rendering state for the ``stc`` command-line interface."""

from __future__ import annotations

from dataclasses import dataclass

import typer
from rich.console import Console

__all__ = ["CliState", "console", "state_from_context", "truncate"]


@dataclass(frozen=True, slots=True)
class CliState:
    """Global options that every command may consult."""

    debug: bool = False
    quiet: bool = False
    log_format: str = "console"


def state_from_context(context: typer.Context | None) -> CliState:
    """Return the callback state, falling back to safe defaults."""
    if context is not None and isinstance(context.obj, CliState):
        return context.obj
    return CliState()


def console(context: typer.Context | None) -> Console:
    """Return a console that honours ``--quiet``."""
    return Console(quiet=state_from_context(context).quiet)


def truncate(text: str, width: int = 60) -> str:
    """Shorten ``text`` to ``width`` characters with an ellipsis."""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"
