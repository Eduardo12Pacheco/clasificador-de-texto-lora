"""``stc bundle``: deterministic release packing and verification."""

from __future__ import annotations

import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.use_cases.pack_bundle import PackBundle
from stc.domain.errors import InvalidInputError

bundle_app = typer.Typer(
    name="bundle",
    help="Pack, verify and inspect model release bundles.",
    no_args_is_help=True,
)


def _asset(name: str) -> str:
    return (files("stc._assets") / name).read_text(encoding="utf-8")


def _notice(manifest: dict[str, object]) -> str:
    base_model = manifest.get("base_model")
    base = base_model if isinstance(base_model, dict) else {}
    template = _asset("NOTICE.bundle.tmpl")
    return template.format(
        base_model=base.get("id", "unknown"),
        base_license=base.get("license", "unknown"),
        version=manifest.get("version", "0.1.0"),
    )


@bundle_app.command("pack")
@handle_domain_errors
def pack(
    bundle: Annotated[
        Path,
        typer.Option("--bundle", help="Bundle directory produced by stc export."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Destination release zip."),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Optional baseline directory to embed."),
    ] = None,
    model_card: Annotated[
        Path | None,
        typer.Option("--model-card", help="Optional MODEL_CARD.md to embed."),
    ] = None,
    update_registry: Annotated[
        Path | None,
        typer.Option("--update-registry", help="Registry JSON to upsert the release entry into."),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option("--url", help="Release asset URL recorded in the registry."),
    ] = None,
) -> None:
    """Assemble the D13 release layout and write a deterministic zip."""
    if (update_registry is None) != (url is None):
        raise InvalidInputError("--update-registry and --url must be given together")
    archive = bootstrap.build_bundle_archive()
    manifest = archive.verify(bundle)
    result = PackBundle(archive).run(
        bundle,
        out,
        baseline_dir=baseline,
        model_card=model_card,
        license_text=_asset("LICENSE-Apache-2.0.txt"),
        notice_text=_notice(dict(manifest)),
    )
    if update_registry is not None and url is not None:
        base_model = manifest.get("base_model")
        base = base_model if isinstance(base_model, dict) else {}
        bootstrap.upsert_registry_entry(
            update_registry,
            name=result.name,
            version=result.version,
            url=url,
            sha256=result.sha256,
            size_bytes=result.size_bytes,
            license=str(base.get("license", "unknown")),
        )
        typer.echo(f"updated registry {update_registry}")
    typer.echo(f"sha256 {result.sha256}")
    typer.echo(f"size_bytes {result.size_bytes}")
    typer.echo(f"wrote {result.path}")


@bundle_app.command("pack-adapter")
@handle_domain_errors
def pack_adapter(
    run: Annotated[
        Path,
        typer.Option("--run", help="Finished training run with an adapter/ directory."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Destination adapter zip."),
    ],
) -> None:
    """Zip the LoRA adapter with run metadata for release."""
    archive = bootstrap.build_bundle_archive()
    with tempfile.TemporaryDirectory() as scratch:
        result = PackBundle(archive).run_adapter(run, out, Path(scratch))
    typer.echo(f"sha256 {result.sha256}")
    typer.echo(f"size_bytes {result.size_bytes}")
    typer.echo(f"wrote {result.path}")


@bundle_app.command("verify")
@handle_domain_errors
def verify(
    path: Annotated[Path, typer.Argument(help="Bundle directory to verify.")],
) -> None:
    """Verify every manifest hash of a bundle directory."""
    manifest = bootstrap.build_bundle_archive().verify(path)
    typer.echo(f"OK {manifest['name']} {manifest['version']} ({manifest['kind']})")


__all__ = ["bundle_app"]
