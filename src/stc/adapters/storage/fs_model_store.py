"""Filesystem model store: install, locate and verify local bundles (D13)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from stc.application.dto import ModelInfo
from stc.domain.errors import ArtifactNotFoundError, InvalidInputError
from stc.domain.labels import LabelSpace

__all__ = ["BundleFiles", "FsModelStore"]


class BundleFiles(Protocol):
    """The bundle helpers the store needs; injected by the composition root."""

    def verify(self, bundle_dir: Path) -> Mapping[str, object]:
        """Verify a bundle and return its manifest."""
        ...

    def read_manifest(self, bundle_dir: Path) -> Mapping[str, object]:
        """Read a manifest without hashing."""
        ...

    def load_labels(self, bundle_dir: Path) -> LabelSpace:
        """Read the bundle label space."""
        ...

    def bundle_info(
        self, bundle_dir: Path, manifest: Mapping[str, object], labels: LabelSpace
    ) -> ModelInfo:
        """Describe a bundle."""
        ...

    def copy_tree(self, source: Path, dest: Path) -> None:
        """Copy a directory tree."""
        ...


def _component(value: str, field_name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise InvalidInputError(f"{field_name} must be a simple path component")
    return value


class FsModelStore:
    """Store bundles under ``<root>/<name>/<version>/``."""

    def __init__(self, root: Path, *, bundle: BundleFiles) -> None:
        self._root = Path(root)
        self._bundle = bundle

    @property
    def root(self) -> Path:
        """Directory every installed bundle lives under."""
        return self._root

    def save(self, bundle_dir: Path, name: str, version: str) -> Path:
        """Verify ``bundle_dir`` and install it as ``name``/``version``."""
        source = Path(bundle_dir)
        self._bundle.verify(source)
        target = self._root / _component(name, "name") / _component(version, "version")
        self._bundle.copy_tree(source, target)
        return target

    def load(self, name: str, version: str) -> Path:
        """Return the directory of an installed bundle."""
        path = self._root / _component(name, "name") / _component(version, "version")
        if not (path / "manifest.json").is_file():
            raise ArtifactNotFoundError(
                f"model {name} {version} is not installed; run: stc model pull {name}"
            )
        return path

    def list_installed(self) -> Sequence[ModelInfo]:
        """List every bundle that carries a readable manifest."""
        installed: list[ModelInfo] = []
        if not self._root.is_dir():
            return tuple(installed)
        for manifest_path in sorted(self._root.glob("*/*/manifest.json")):
            bundle = manifest_path.parent
            try:
                manifest = self._bundle.read_manifest(bundle)
                labels = self._bundle.load_labels(bundle)
            except Exception:  # noqa: S112 - a half-installed bundle must not break the listing
                continue
            installed.append(self._bundle.bundle_info(bundle, manifest, labels))
        return tuple(installed)

    def verify(self, bundle_dir: Path) -> None:
        """Validate a bundle directory without installing it."""
        self._bundle.verify(Path(bundle_dir))
