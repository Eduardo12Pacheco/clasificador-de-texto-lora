"""``stc report``: audited final evaluation and results rendering."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.adapters.cli.rendering import console
from stc.application.ports import DatasetRepository, ReportWriter
from stc.application.use_cases.final_report import FinalReport
from stc.domain.access import TestSplitGuard
from stc.domain.errors import InvalidInputError

report_app = typer.Typer(
    name="report",
    help="Compose the audited final report and render its results.",
    no_args_is_help=True,
)

_DATASETS = {"massive", "fixture"}


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _access_log_path() -> Path:
    return Path("reports/test_access_log.jsonl")


def _repository(dataset: str) -> DatasetRepository:
    if dataset not in _DATASETS:
        raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")
    services = bootstrap.build_data_services()
    return services.fixtures if dataset == "fixture" else services.repository


@report_app.command("final")
@handle_domain_errors
def final(
    ctx: typer.Context,
    bundle: Annotated[
        Path,
        typer.Option("--bundle", help="Shipped ONNX bundle directory."),
    ],
    baseline: Annotated[
        Path,
        typer.Option("--baseline", help="Fitted baseline directory."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the final report files."),
    ],
    figures: Annotated[
        Path,
        typer.Option("--figures", help="Directory for the report figures."),
    ],
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
) -> None:
    """Read the locked test split once and write every audited report artifact."""
    settings = bootstrap.build_settings()
    model_artifact = bootstrap.load_calibrated_classifier(str(bundle), settings)
    baseline_artifact = bootstrap.load_calibrated_classifier(str(baseline), settings)
    repository = _repository(dataset)
    command = f"stc report final --bundle {bundle} --baseline {baseline} --dataset {dataset}"
    guard = TestSplitGuard()
    grant = guard.grant(
        "final report",
        model_artifact.classifier.info.model_id,
        _SystemClock(),
        model_sha256=bootstrap.artifact_sha256(model_artifact.path),
    )
    access_log = bootstrap.build_access_log(_access_log_path())
    writer: ReportWriter = bootstrap.build_report_writer()
    use_case = FinalReport(
        repository,
        writer,
        access_log,
        figures=bootstrap.build_final_report_figures(),
    )
    result = use_case.run(
        baseline_artifact.classifier,
        model_artifact.classifier,
        {
            "baseline": baseline_artifact.calibration,
            "lora": model_artifact.calibration,
        },
        model_artifact.classifier.labels,
        grant,
        out,
        figures,
        command=command,
        git_sha=_git_sha(),
    )
    table = Table(title="Final report (test split)")
    table.add_column("model")
    table.add_column("accuracy", justify="right")
    table.add_column("macro-F1", justify="right")
    for key in ("baseline", "lora"):
        payload = _model_payload(out, key)
        table.add_row(
            key,
            f"{_number(payload, 'accuracy'):.4f}",
            f"{_number(payload, 'macro_f1'):.4f}",
        )
    console(ctx).print(table)
    typer.echo(f"verdict: {result.verdict}")
    typer.echo(result.verdict_sentences["en"])
    typer.echo(f"wrote {out}")


def _number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInputError(f"metrics.json has no numeric {key}")
    return float(value)


def _model_payload(out: Path, key: str) -> dict[str, object]:
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    models = payload["models"]
    if not isinstance(models, dict):
        raise InvalidInputError("metrics.json has no models object")
    block = models[key]
    if not isinstance(block, dict):
        raise InvalidInputError(f"metrics.json has no {key} model block")
    return block


@report_app.command("render")
@handle_domain_errors
def render(
    summary: Annotated[
        Path,
        typer.Option("--summary", help="Summary JSON path (rebuilt when stale)."),
    ] = Path("reports/summary.json"),
    table: Annotated[
        Path,
        typer.Option("--table", help="Markdown results table to write."),
    ] = Path("reports/results_table.md"),
    readme: Annotated[
        Path | None,
        typer.Option("--readme", help="English README whose RESULTS block is replaced."),
    ] = None,
    readme_es: Annotated[
        Path | None,
        typer.Option("--readme-es", help="Spanish README whose RESULTS block is replaced."),
    ] = None,
    final_dir: Annotated[
        Path | None,
        typer.Option("--final-dir", help="Final-report directory (default: sibling of --summary)."),
    ] = None,
) -> None:
    """Rebuild the summary when needed and render the table and README blocks."""
    try:
        bootstrap.render_results(
            summary_path=summary,
            table_path=table,
            final_dir=final_dir,
            readme=readme,
            readme_es=readme_es,
        )
    except (OSError, ValueError) as error:
        raise InvalidInputError(str(error)) from error
    typer.echo(f"wrote {summary} and {table}")
