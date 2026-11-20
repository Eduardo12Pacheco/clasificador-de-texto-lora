"""``stc predict``: calibrated intent predictions from args, a file or stdin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.table import Table

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.adapters.cli.rendering import console, truncate
from stc.application.use_cases.predict_texts import PredictTexts
from stc.domain.errors import InvalidInputError


def _read_file(path: Path) -> list[str]:
    if str(path) == "-":
        text = sys.stdin.read()
    else:
        if not path.is_file():
            raise InvalidInputError(f"input file not found: {path}")
        text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _texts_from_stdin() -> list[str]:
    return [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]


def _collect_texts(texts: list[str], source: Path | None) -> list[str]:
    if texts and texts != ["-"]:
        return [text for text in texts if text.strip()]
    if texts == ["-"]:
        return _texts_from_stdin()
    if source is not None:
        return _read_file(source)
    if not sys.stdin.isatty():
        return _texts_from_stdin()
    return []


@handle_domain_errors
def predict(
    ctx: typer.Context,
    texts: Annotated[
        list[str] | None,
        typer.Argument(help="Texts to classify; use - to read stdin."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Read one text per line ('-' reads stdin)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name, path or 'baseline' (default: STC_MODEL)."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of calibrated candidates per text."),
    ] = 3,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the schema-versioned prediction document."),
    ] = False,
) -> None:
    """Classify Spanish virtual-assistant requests and flag uncertain ones."""
    if top_k < 1:
        raise InvalidInputError("--top-k must be a positive integer")
    selected = [text for text in (texts or []) if text.strip()]
    items = _collect_texts(selected, file)
    if not items:
        raise InvalidInputError("no input text; pass a text argument, --file or pipe stdin")

    settings = bootstrap.build_settings()
    artifact = bootstrap.load_calibrated_classifier(model or settings.model, settings)
    payload = PredictTexts().run(artifact.classifier, items, top_k=top_k)

    if as_json:
        typer.echo(bootstrap.build_report_writer().dumps(payload))
        return

    model_block = cast(dict[str, object], payload["model"])
    predictions = cast(list[dict[str, object]], payload["predictions"])
    table = Table(title=f"Predictions ({model_block['id']})")
    table.add_column("text")
    table.add_column("intent")
    table.add_column("scenario")
    table.add_column("confidence", justify="right")
    table.add_column("review")
    for prediction in predictions:
        index = int(cast(int, prediction["index"]))
        table.add_row(
            truncate(items[index]),
            str(prediction["intent"]),
            str(prediction["scenario"]),
            f"{float(cast(float, prediction['confidence'])):.3f}",
            "yes" if prediction["needs_review"] else "no",
        )
    console(ctx).print(table)
