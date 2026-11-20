"""``stc train``: fit the classical baseline (transformer training arrives in T7)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.use_cases.train_baseline import load_baseline_grid
from stc.domain.errors import InvalidInputError

_DATASETS = {"massive", "fixture"}

train_app = typer.Typer(
    name="train",
    help="Train the classical baseline or a transformer classifier.",
    no_args_is_help=True,
)


def _format_params(params: dict[str, object]) -> str:
    parts = [f"classifier={params.get('classifier')}"]
    for key in ("word_ngram", "char_ngram", "C", "class_weight"):
        if key in params:
            parts.append(f"{key}={params[key]}")
    return " ".join(parts)


@train_app.command("baseline")
@handle_domain_errors
def train_baseline(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the fitted baseline artifacts."),
    ] = Path("artifacts/baseline"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="TOML grid file (defaults to configs/baseline.toml when present).",
        ),
    ] = None,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Fit only the first two SVC configs (smoke tests)."),
    ] = False,
    reports: Annotated[
        Path | None,
        typer.Option(
            "--reports",
            help="Directory for search.json and dev_metrics.json (defaults to --out).",
        ),
    ] = None,
) -> None:
    """Grid-tune the TF-IDF + linear baseline on dev and persist the winner."""
    if dataset not in _DATASETS:
        raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")
    grid = load_baseline_grid(config)
    services = bootstrap.build_baseline_services(dataset=dataset)
    result = services.use_case.run(
        services.repository,
        out_dir=out,
        grid=grid,
        quick=quick,
        reports_dir=reports,
    )
    winner = dict(result.winner.params)
    macro_f1 = float(result.dev_metrics["macro_f1"])  # type: ignore[arg-type]
    accuracy = float(result.dev_metrics["accuracy"])  # type: ignore[arg-type]
    typer.echo(f"winner: {_format_params(winner)}")
    typer.echo(f"dev macro_f1={macro_f1:.4f} accuracy={accuracy:.4f} n_configs={result.n_configs}")
    typer.echo(f"wrote {result.artifact_dir} and {result.search_path}")
