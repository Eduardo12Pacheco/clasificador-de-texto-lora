"""``stc data`` commands: fetch, verify, prepare, audit and fixture info."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.use_cases.audit_data import AuditDataUseCase
from stc.domain.splits import Split

data_app = typer.Typer(
    name="data",
    help="Download, verify, prepare and audit the MASSIVE dataset.",
    no_args_is_help=True,
)
fixture_app = typer.Typer(
    name="fixture",
    help="Inspect the CC BY 4.0 sample bundled with the package.",
    no_args_is_help=True,
)
data_app.add_typer(fixture_app, name="fixture")


@data_app.command("fetch")
@handle_domain_errors
def fetch(
    url: Annotated[
        str | None,
        typer.Option("--url", help="Override the pinned archive URL (tests and mirrors)."),
    ] = None,
    sha256: Annotated[
        str | None,
        typer.Option("--sha256", help="Override the expected archive sha256."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download even when a verified cache exists."),
    ] = False,
) -> None:
    """Download the pinned archive, verify size and sha256, then extract it."""
    services = bootstrap.build_data_services(url=url, sha256=sha256)
    ref = services.source.fetch(services.raw_dir, force=force)
    if ref.cached:
        typer.echo(f"cache hit {ref.root}")
    else:
        typer.echo(f"sha256 OK {ref.sha256}")
        typer.echo(f"extracted to {ref.root}")


@data_app.command("verify")
@handle_domain_errors
def verify() -> None:
    """Re-hash the cached archive and confirm it still matches the pin."""
    services = bootstrap.build_data_services()
    digest = services.source.verify(services.raw_dir)
    typer.echo(f"verified {digest}")


@data_app.command("prepare")
@handle_domain_errors
def prepare() -> None:
    """Clean the train split, keep dev/test intact and persist all three."""
    services = bootstrap.build_data_services()
    result = services.pipeline.run()
    counts = result.meta.processed_counts
    dropped = sum(result.meta.dropped_counts.values())
    typer.echo(
        f"prepared train={counts.get('train', 0)} dev={counts.get('dev', 0)} "
        f"test={counts.get('test', 0)} dropped={dropped}"
    )


@data_app.command("audit")
@handle_domain_errors
def audit(
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for data_audit.json and data_audit.md."),
    ] = Path("reports"),
) -> None:
    """Write the leakage and duplication audit report."""
    services = bootstrap.build_data_services()
    paths = AuditDataUseCase(pipeline=services.pipeline, out_dir=out).run()
    typer.echo(f"wrote {paths.json_path}")
    typer.echo(f"wrote {paths.markdown_path}")


@fixture_app.command("info")
@handle_domain_errors
def fixture_info() -> None:
    """Print the row counts and size of the bundled fixture."""
    fixtures = bootstrap.build_data_services().fixtures
    counts = {split.value: len(fixtures.load(split)) for split in Split}
    total = sum(counts.values())
    typer.echo(f"train {counts['train']}")
    typer.echo(f"dev {counts['dev']}")
    typer.echo(f"test {counts['test']}")
    typer.echo(f"total {total} rows, {fixtures.source_size_bytes()} bytes")
