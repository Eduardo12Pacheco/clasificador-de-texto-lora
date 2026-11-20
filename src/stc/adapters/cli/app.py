"""Typer entrypoint for the ``stc`` command-line interface."""

from typing import Annotated

import typer

from stc import __version__
from stc.adapters.cli.commands.benchmark import benchmark
from stc.adapters.cli.commands.bundle import bundle_app
from stc.adapters.cli.commands.calibrate import calibrate
from stc.adapters.cli.commands.data import data_app
from stc.adapters.cli.commands.doctor import doctor
from stc.adapters.cli.commands.evaluate import evaluate
from stc.adapters.cli.commands.export import export
from stc.adapters.cli.commands.model import model_app
from stc.adapters.cli.commands.train import train_app
from stc.domain.errors import DomainError

APP_NAME = "stc"

app = typer.Typer(
    name=APP_NAME,
    help="Spanish virtual-assistant intent classifier.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(bundle_app, name="bundle")
app.add_typer(model_app, name="model")
app.command("evaluate")(evaluate)
app.command("export")(export)
app.command("calibrate")(calibrate)
app.command("benchmark")(benchmark)
app.command("doctor")(doctor)


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
    try:
        app()
    except DomainError as error:
        typer.echo(f"error: {error.message}", err=True)
        raise typer.Exit(code=error.exit_code) from error
