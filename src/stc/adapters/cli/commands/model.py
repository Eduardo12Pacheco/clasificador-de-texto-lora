"""``stc model``: install, list, inspect and verify local model bundles."""

from __future__ import annotations

import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.use_cases.pull_model import PullModel
from stc.domain.errors import ArtifactNotFoundError, OfflineError

model_app = typer.Typer(
    name="model",
    help="Install, list and verify model bundles.",
    no_args_is_help=True,
)


def _default_registry() -> Path:
    return Path(str(files("stc") / "model_registry.json"))


def _installed_version(name: str) -> str:
    versions = sorted(
        info.version
        for info in bootstrap.build_model_store().list_installed()
        if info.model_id == name
    )
    if not versions:
        raise ArtifactNotFoundError(f"model {name} is not installed; run: stc model pull {name}")
    return versions[-1]


@model_app.command("pull")
@handle_domain_errors
def pull(
    name: Annotated[
        str | None,
        typer.Argument(help="Model name to install from the registry."),
    ] = None,
    from_file: Annotated[
        Path | None,
        typer.Option("--from-file", help="Install a local bundle zip instead of downloading."),
    ] = None,
    version: Annotated[
        str | None,
        typer.Option("--version", help="Registry version to install (latest by default)."),
    ] = None,
    registry: Annotated[
        Path | None,
        typer.Option("--registry", help="Registry JSON path (defaults to the packaged one)."),
    ] = None,
) -> None:
    """Install a bundle from the registry or a local zip, verifying sha256."""
    settings = bootstrap.build_settings()
    if from_file is None and settings.offline:
        raise OfflineError("network access is disabled by STC_OFFLINE")
    store = bootstrap.build_model_store()
    puller = PullModel(
        bootstrap.build_model_hub(),
        store,
        bootstrap.build_bundle_archive(),
    )
    with tempfile.TemporaryDirectory() as scratch:
        if from_file is not None:
            result = puller.run_from_file(from_file, Path(scratch))
        else:
            selected = registry if registry is not None else _default_registry()
            resolved = bootstrap.build_registry(selected).find(name or settings.model, version)
            result = puller.run_entry(
                {
                    "url": resolved.url,
                    "sha256": resolved.sha256,
                    "size_bytes": resolved.size_bytes,
                },
                Path(scratch),
            )
    typer.echo(f"installed {result.name} {result.version} at {result.path}")


@model_app.command("list")
@handle_domain_errors
def list_models() -> None:
    """List every installed model bundle."""
    installed = bootstrap.build_model_store().list_installed()
    table = Table(title="installed models")
    table.add_column("name")
    table.add_column("version")
    table.add_column("kind")
    table.add_column("quantization")
    for info in installed:
        table.add_row(
            info.model_id,
            info.version,
            info.kind,
            info.quantization or "none",
        )
    Console().print(table)


@model_app.command("info")
@handle_domain_errors
def info(
    name: Annotated[
        str | None,
        typer.Argument(help="Model name (all installed models when omitted)."),
    ] = None,
) -> None:
    """Show metadata of installed bundles."""
    installed = bootstrap.build_model_store().list_installed()
    if name is not None:
        installed = tuple(info for info in installed if info.model_id == name)
        if not installed:
            raise ArtifactNotFoundError(
                f"model {name} is not installed; run: stc model pull {name}"
            )
    for entry in installed:
        typer.echo(f"{entry.model_id} {entry.version}")
        typer.echo(f"  kind={entry.kind} quantization={entry.quantization or 'none'}")
        typer.echo(
            f"  parameters={entry.parameters} license={entry.license} "
            f"intents={entry.n_intents} scenarios={entry.n_scenarios}"
        )


@model_app.command("verify")
@handle_domain_errors
def verify(
    target: Annotated[
        str | None,
        typer.Argument(help="Installed model name or bundle directory (default: STC_MODEL)."),
    ] = None,
) -> None:
    """Verify a bundle directory or an installed model's manifest and hashes."""
    settings = bootstrap.build_settings()
    selected = target or settings.model
    path = Path(selected)
    if not path.is_dir():
        store = bootstrap.build_model_store()
        path = store.load(selected, _installed_version(selected))
    manifest = bootstrap.build_bundle_archive().verify(path)
    typer.echo(f"OK {path} ({manifest['name']} {manifest['version']})")
