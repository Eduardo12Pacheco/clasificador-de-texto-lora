"""``stc benchmark``: measure CPU single-utterance latency and batch throughput."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from stc import bootstrap
from stc.adapters.cli.commands.doctor import _cpu_model
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.use_cases.benchmark_model import BenchmarkModel
from stc.domain.errors import InvalidInputError
from stc.domain.splits import Split


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise InvalidInputError(f"{name} must be a comma-separated list of integers") from error
    if not parsed or any(item < 1 for item in parsed):
        raise InvalidInputError(f"{name} must contain positive integers")
    return parsed


def _onnxruntime_version() -> str:
    try:
        return importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


@handle_domain_errors
def benchmark(
    bundle: Annotated[
        Path,
        typer.Option("--bundle", help="Exported model bundle directory."),
    ],
    threads: Annotated[
        str,
        typer.Option("--threads", help="Comma-separated ORT intra-op thread counts, e.g. 1,4."),
    ] = "1,4",
    runs: Annotated[
        int,
        typer.Option("--runs", help="Single-utterance measurements per thread count."),
    ] = 200,
    out: Annotated[
        Path,
        typer.Option("--out", help="Destination benchmark.json."),
    ] = Path("reports/final/benchmark.json"),
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset whose dev texts are benchmarked."),
    ] = "massive",
) -> None:
    """Record p50/p95 latency per thread count and batch-32 throughput."""
    if runs < 1:
        raise InvalidInputError("--runs must be positive")
    counts = _parse_ints(threads, "--threads")
    repository = bootstrap.build_training_repository(dataset=dataset)
    texts = [example.text for example in repository.load(Split.DEV)]
    if not texts:
        raise InvalidInputError("benchmark needs a non-empty dev split")
    loader = bootstrap.build_onnx_loader()
    payload = BenchmarkModel().run(
        lambda count: loader.load(bundle, threads=count),
        texts,
        threads=counts,
        runs=runs,
        environment={
            "cpu_model": _cpu_model(),
            "onnxruntime_version": _onnxruntime_version(),
        },
    )
    bootstrap.build_report_writer().write_json(out, payload)
    table = Table(title="CPU benchmark")
    table.add_column("threads", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("batch32/s", justify="right")
    rows = payload["threads"]
    for row in rows if isinstance(rows, list) else []:
        table.add_row(
            str(row["threads"]),
            f"{float(row['p50_ms']):.2f}",
            f"{float(row['p95_ms']):.2f}",
            f"{float(row['batch32_throughput_per_s']):.1f}",
        )
    Console().print(table)
    typer.echo(f"wrote {out}")
