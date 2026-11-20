"""Install a model bundle from a release entry or a local zip (D13)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from stc.application.dto import PullResult
from stc.application.ports import BundleArchive, ModelHub, ModelStore
from stc.domain.errors import IntegrityError

__all__ = ["PullModel"]


class PullModel:
    """Verify, extract and install one bundle."""

    def __init__(self, hub: ModelHub, store: ModelStore, archive: BundleArchive) -> None:
        self._hub = hub
        self._store = store
        self._archive = archive

    def run_entry(self, entry: Mapping[str, object], scratch_dir: Path) -> PullResult:
        """Pull a registry entry and install the bundle it contains."""
        extracted = self._hub.pull(entry, scratch_dir)
        manifest = self._archive.verify(extracted)
        return self._install(extracted, manifest)

    def run_from_file(self, archive: Path, scratch_dir: Path) -> PullResult:
        """Install a bundle from a local zip, verifying every hash first."""
        extracted = self._archive.unpack(Path(archive), Path(scratch_dir) / "extracted")
        manifest = self._archive.verify(extracted)
        return self._install(extracted, manifest)

    def _install(self, bundle_dir: Path, manifest: Mapping[str, object]) -> PullResult:
        name = manifest.get("name")
        version = manifest.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise IntegrityError("bundle manifest needs a name and version")
        path = self._store.save(bundle_dir, name, version)
        return PullResult(path=path, name=name, version=version)
