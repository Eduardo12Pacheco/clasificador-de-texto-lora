"""``stc export``: merge a run, export ONNX and evaluate the D12 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.application.use_cases.export_model import ExportModel
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.splits import Split

_DATASETS = {"massive", "fixture"}


def _repository(dataset: str) -> DevOnlyRepository:
    if dataset not in _DATASETS:
        raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")
    return DevOnlyRepository(bootstrap.build_training_repository(dataset=dataset))


def _dev_examples(dataset: str) -> list[Example]:
    return list(_repository(dataset).load(Split.DEV))


def _calibration_texts(dataset: str) -> list[str]:
    train = sorted(_repository(dataset).load(Split.TRAIN), key=lambda row: int(row.id))
    return [row.text for row in train[:512]]


@handle_domain_errors
def export(
    run: Annotated[
        Path,
        typer.Option("--run", help="Finished training run directory (contains run.json)."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the exported ONNX bundle."),
    ] = Path("artifacts/bundle"),
    quantize: Annotated[
        str,
        typer.Option("--quantize", help="Quantisation matrix: none (default) or all."),
    ] = "none",
    reports: Annotated[
        Path | None,
        typer.Option(
            "--reports",
            help="Directory for export_parity.json and quantisation results.",
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset for the dev parity/Gate evaluation."),
    ] = "massive",
) -> None:
    """Merge a trained run, export ONNX (opset 17) and measure parity and int8 quality."""
    if quantize not in {"none", "all"}:
        raise InvalidInputError("--quantize must be 'none' or 'all'")
    dev = _dev_examples(dataset)
    calibration = _calibration_texts(dataset) if quantize == "all" else None
    use_case = ExportModel(
        bootstrap.build_exporter(),
        bootstrap.build_onnx_loader(),
        bootstrap.build_report_writer(),
    )
    outcome = use_case.run(
        run,
        out,
        dev=dev,
        quantize=quantize,
        reports_dir=reports,
        calibration_texts=calibration,
    )
    parity = outcome.parity
    typer.echo(
        f"parity max_abs_logit_diff={parity['max_abs_logit_diff']} "
        f"agreement={parity['agreement']} passed={parity['passed']}"
    )
    typer.echo(f"shipped_format={outcome.shipped_format}")
    typer.echo(f"wrote {outcome.bundle_dir}")
