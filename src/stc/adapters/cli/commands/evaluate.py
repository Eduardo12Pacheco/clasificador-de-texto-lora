"""``stc evaluate``: audited evaluation of a classifier on dev or test."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.ports import Classifier
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.application.use_cases.evaluate_model import EvaluateModel
from stc.application.use_cases.robustness import RobustnessEvaluator, RobustnessReport
from stc.domain.access import LOCKED_MESSAGE, TestSplitGuard
from stc.domain.errors import InvalidInputError, TestSplitLocked
from stc.domain.examples import Example
from stc.domain.perturbations import PERTURBATIONS
from stc.domain.splits import Split

_DATASETS = {"massive", "fixture"}


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def _git_sha() -> str:
    try:
        # The argument list is fixed; no shell and no user-controlled executable.
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


def _load_examples(dataset: str, split: Split) -> Sequence[Example]:
    if dataset not in _DATASETS:
        raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")
    services = bootstrap.build_data_services()
    repository = services.fixtures if dataset == "fixture" else services.repository
    if split is Split.TEST:
        return repository.load(split)
    return DevOnlyRepository(repository).load(split)


def _load_classifier(model: str) -> Classifier:
    return bootstrap.load_classifier(model)


@handle_domain_errors
def evaluate(
    model: Annotated[
        str,
        typer.Option("--model", help="Path to a baseline directory or a registry name."),
    ],
    split: Annotated[
        Split,
        typer.Option("--split", help="Split to evaluate: dev (default) or test."),
    ] = Split.DEV,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Directory for evaluation.json."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the evaluation payload as JSON."),
    ] = False,
    final: Annotated[
        bool,
        typer.Option(
            "--final",
            help="Authorise and audit one read of the locked test split.",
        ),
    ] = False,
    robustness: Annotated[
        bool,
        typer.Option(
            "--robustness",
            help="Append the S9 perturbation table (dev only unless --final).",
        ),
    ] = False,
) -> None:
    """Evaluate a classifier and record every metric in a deterministic report."""
    if split is Split.TRAIN:
        raise InvalidInputError("evaluation supports dev or test; train is not evaluated")
    if split is Split.TEST and not final:
        raise TestSplitLocked(LOCKED_MESSAGE)

    classifier = _load_classifier(model)
    examples = _load_examples(dataset, split)
    grant = None
    access_log = None
    command = f"stc evaluate --split {split.value} --model {model}"
    if split is Split.TEST:
        command += " --final"
        grant = TestSplitGuard().grant("final evaluation", classifier.info.model_id, _SystemClock())
        access_log = bootstrap.build_access_log(_access_log_path())

    result = EvaluateModel().run(
        classifier,
        examples,
        split=split,
        grant=grant,
        access_log=access_log,
        command=command,
        git_sha=_git_sha(),
    )
    payload = result.to_dict()
    robustness_report = None
    if robustness:
        robustness_report = RobustnessEvaluator().run(
            classifier,
            examples,
            list(PERTURBATIONS),
            split=split,
        )
        payload["robustness"] = robustness_report.to_dict()
    writer = bootstrap.build_report_writer()
    if out is not None:
        writer.write_json(out / "evaluation.json", payload)
        if robustness_report is not None:
            writer.write_json(out / "robustness.json", robustness_report.to_dict())
    if as_json:
        typer.echo(writer.dumps(payload))
    else:
        typer.echo(
            f"split={result.split.value} n={result.n_examples} "
            f"accuracy={payload['accuracy']} macro_f1={payload['macro_f1']} "
            f"top3={payload['top3_accuracy']} ece={payload['ece']}"
        )
        if robustness_report is not None:
            _print_robustness(robustness_report)
        if out is not None:
            typer.echo(f"wrote {out / 'evaluation.json'}")


def _print_robustness(report: RobustnessReport) -> None:
    table = Table(title="Robustness (changed subset)")
    table.add_column("perturbation")
    table.add_column("affected", justify="right")
    table.add_column("accuracy before", justify="right")
    table.add_column("accuracy after", justify="right")
    table.add_column("delta", justify="right")
    for row in report.perturbations:
        table.add_row(
            row.name,
            str(row.n_affected),
            f"{row.accuracy_before:.3f}",
            f"{row.accuracy_after:.3f}",
            f"{row.delta_accuracy:+.3f}",
        )
    Console().print(table)
