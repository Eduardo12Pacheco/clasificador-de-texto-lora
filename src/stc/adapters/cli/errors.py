"""Map expected domain errors to stable CLI exit codes."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import typer

from stc.domain.errors import DomainError


def handle_domain_errors[**P, R](command: Callable[P, R]) -> Callable[P, R]:
    """Print a one-line error and exit with the domain error's stable code."""

    @wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except DomainError as error:
            typer.echo(f"error: {error.message}", err=True)
            raise typer.Exit(code=error.exit_code) from error

    return wrapper
