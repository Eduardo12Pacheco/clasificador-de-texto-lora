"""Assemble deterministic release bundles (D13)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from stc.application.ports import BundleArchive

__all__ = ["PackAdapterResult", "PackBundle", "PackResult", "compute_sha256"]

MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PackResult:
    """A written release archive."""

    path: Path
    sha256: str
    size_bytes: int
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class PackAdapterResult:
    """A written LoRA-adapter archive."""

    path: Path
    sha256: str
    size_bytes: int


class PackBundle:
    """Build the D13 release layout and zip it deterministically."""

    def __init__(self, archive: BundleArchive) -> None:
        self._archive = archive

    def run(
        self,
        bundle_dir: Path,
        out: Path,
        *,
        baseline_dir: Path | None = None,
        model_card: Path | None = None,
        license_text: str,
        notice_text: str,
    ) -> PackResult:
        """Assemble ``bundle_dir`` plus release companions and write ``out``."""
        root = Path(bundle_dir)
        manifest = self._archive.verify(root)
        name = str(manifest["name"])
        version = str(manifest["version"])
        with tempfile.TemporaryDirectory() as scratch:
            staging = Path(scratch) / "bundle"
            staging.mkdir(parents=True)
            for source in sorted(root.rglob("*")):
                if not source.is_file() or source.name == MANIFEST_NAME:
                    continue
                relative = source.relative_to(root)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if baseline_dir is not None:
                shutil.copytree(baseline_dir, staging / "baseline")
            if model_card is not None:
                shutil.copyfile(model_card, staging / "MODEL_CARD.md")
            (staging / "LICENSE").write_text(license_text, encoding="utf-8", newline="\n")
            (staging / "NOTICE").write_text(notice_text, encoding="utf-8", newline="\n")
            payload = dict(manifest)
            payload["files"] = [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ]
            (staging / MANIFEST_NAME).write_text(_dump(payload), encoding="utf-8", newline="\n")
            target_zip = self._archive.write_zip(staging, Path(out))
        digest = _sha256(target_zip)
        return PackResult(
            path=target_zip,
            sha256=digest,
            size_bytes=target_zip.stat().st_size,
            name=name,
            version=version,
        )

    def run_adapter(self, run_dir: Path, out: Path, scratch_dir: Path) -> PackAdapterResult:
        """Zip the LoRA adapter, ``run.json`` and its tokenizer for release."""
        source = Path(run_dir)
        adapter = source / "adapter"
        if not adapter.is_dir():
            raise FileNotFoundError(f"adapter directory not found at {adapter}")
        staging = Path(scratch_dir) / "adapter-bundle"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copytree(adapter, staging / "adapter")
        for name in ("run.json", "metrics.json", "tokenizer.json"):
            candidate = source / name
            if candidate.is_file():
                shutil.copyfile(candidate, staging / name)
        target = self._archive.write_zip(staging, Path(out))
        return PackAdapterResult(
            path=target,
            sha256=_sha256(target),
            size_bytes=target.stat().st_size,
        )


def _dump(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def compute_sha256(path: Path) -> str:
    """Public alias used by the CLI to print archive digests."""
    return _sha256(Path(path))
