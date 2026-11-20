"""Typer entrypoint for the ``stc`` command-line interface."""

from typing import Annotated, Literal, cast

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
from stc.adapters.cli.commands.predict import predict
from stc.adapters.cli.commands.serve import serve
from stc.adapters.cli.commands.train import train_app
from stc.adapters.cli.rendering import CliState
from stc.config import Settings
from stc.domain.errors import DomainError
from stc.logging import configure_logging

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
app.command("predict")(predict)
app.command("serve")(serve)

_DEBUG = False


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show tracebacks for unexpected failures."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress tables and non-essential output."),
    ] = False,
    log_format: Annotated[
        str | None,
        typer.Option("--log-format", help="Structured log format: console or json."),
    ] = None,
) -> None:
    """Classify Spanish virtual-assistant requests locally."""
    global _DEBUG
    if log_format is not None and log_format not in {"console", "json"}:
        raise typer.BadParameter("--log-format must be 'console' or 'json'")
    _DEBUG = debug
    settings = Settings()
    resolved_format = log_format if log_format is not None else settings.resolve_log_format()
    configure_logging(settings.log_level, cast(Literal["console", "json"], resolved_format))
    ctx.obj = CliState(debug=debug, quiet=quiet, log_format=resolved_format)


def main() -> None:
    """Run the command-line application."""
    try:
        app()
    except typer.Exit:
        raise
    except DomainError as error:
        typer.echo(f"error: {error.message}", err=True)
        raise typer.Exit(code=error.exit_code) from error
    except Exception as error:
        if _DEBUG:
            raise
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
