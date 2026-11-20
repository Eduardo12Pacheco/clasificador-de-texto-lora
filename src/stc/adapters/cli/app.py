"""Typer entrypoint for the ``stc`` command-line interface."""

from typing import Annotated

import typer

from stc import __version__

APP_NAME = "stc"

app = typer.Typer(
    name=APP_NAME,
    help="Spanish virtual-assistant intent classifier.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Classify Spanish virtual-assistant requests locally."""


def main() -> None:
    """Run the command-line application."""
    app()
