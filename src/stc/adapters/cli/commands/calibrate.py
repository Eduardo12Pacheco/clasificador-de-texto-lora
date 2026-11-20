"""``stc calibrate``: fit temperature scaling and the review threshold on dev."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.application.use_cases.calibrate_model import CalibrateModel, CalibrationReport
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.splits import Split

_DATASETS = {"massive", "fixture"}


def _dev_examples(dataset: str) -> list[Example]:
    if dataset not in _DATASETS:
        raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")
    repository = DevOnlyRepository(bootstrap.build_training_repository(dataset=dataset))
    return list(repository.load(Split.DEV))


def _fit(path: Path, examples: list[Example]) -> CalibrationReport:
    classifier = bootstrap.load_classifier(str(path))
    return CalibrateModel().run(classifier, examples)


def _write_calibration(target: Path, report: CalibrationReport) -> Path:
    """Write ``calibration.json`` next to a bundle or baseline directory."""
    path = target / "calibration.json"
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


@handle_domain_errors
def calibrate(
    bundle: Annotated[
        Path,
        typer.Option("--bundle", help="Exported model bundle directory."),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Optional baseline directory to calibrate too."),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset used to fit on dev."),
    ] = "massive",
    reports: Annotated[
        Path,
        typer.Option("--reports", help="Directory for calibration_dev.json."),
    ] = Path("reports/final"),
) -> None:
    """Fit D10 temperature scaling and the review threshold on dev scores."""
    examples = _dev_examples(dataset)
    bundle_report = _fit(bundle, examples)
    _write_calibration(bundle, bundle_report)
    bootstrap.build_bundle_archive().refresh_manifest(bundle)
    models: dict[str, object] = {"lora": bundle_report.to_dict()}
    if baseline is not None:
        baseline_report = _fit(baseline, examples)
        _write_calibration(baseline, baseline_report)
        models["baseline"] = baseline_report.to_dict()
        typer.echo(
            f"baseline temperature={baseline_report.calibration.temperature:.4f} "
            f"review_threshold={baseline_report.calibration.review_threshold:.4f}"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "split": "dev",
        "models": models,
    }
    writer = bootstrap.build_report_writer()
    writer.write_json(reports / "calibration_dev.json", payload)
    typer.echo(
        f"lora temperature={bundle_report.calibration.temperature:.4f} "
        f"review_threshold={bundle_report.calibration.review_threshold:.4f}"
    )
    typer.echo(f"wrote {reports / 'calibration_dev.json'}")
