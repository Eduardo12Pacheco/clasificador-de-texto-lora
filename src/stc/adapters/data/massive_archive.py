"""Pinned Amazon MASSIVE 1.0 archive: download, verify, extract, cache.

The archive is the only network input of the project (spec S1). The filename
allow-list and the path checks keep a compromised mirror from writing outside
the destination directory.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from stc.application.dto import RawDatasetRef
from stc.domain.errors import (
    ArtifactNotFoundError,
    ChecksumMismatchError,
    DomainError,
    IntegrityError,
    OfflineError,
)

PINNED_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
PINNED_SHA256 = "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
PINNED_SIZE = 39_500_415

#: Archive member name -> extracted file name inside the destination directory.
ALLOWED_MEMBERS: dict[str, str] = {
    "1.0/data/es-ES.jsonl": "es-ES.jsonl",
    "1.0/LICENSE": "LICENSE",
    "1.0/NOTICE.md": "NOTICE.md",
}

_CHUNK_SIZE = 1 << 20
_VERIFIED_FILE = "VERIFIED"


class MassiveArchiveSource:
    """Stream the pinned tarball to disk only if size and sha256 match."""

    def __init__(
        self,
        url: str = PINNED_URL,
        sha256: str = PINNED_SHA256,
        size: int = PINNED_SIZE,
        *,
        offline: bool = False,
        timeout: float = 60.0,
    ) -> None:
        self.url = url
        self.sha256 = sha256
        self.size = size
        self.offline = offline
        self.timeout = timeout

    def fetch(self, dest: Path, *, force: bool = False) -> RawDatasetRef:
        """Return a verified dataset at ``dest``, downloading it when needed."""
        destination = Path(dest)
        if not force and self._is_cached(destination):
            return self._ref(destination, cached=True)
        if self.offline:
            raise OfflineError(
                "offline mode is enabled (STC_OFFLINE=1) and the dataset is not cached; "
                "run without STC_OFFLINE to download it"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        archive = destination.parent / f"{destination.name}.tar.gz"
        part = archive.with_name(archive.name + ".part")
        try:
            self._download(part)
            self._extract(part, destination)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(archive)
        return self._ref(destination, cached=False)

    def verify(self, dest: Path) -> str:
        """Re-hash the cached archive and confirm it still matches the pin."""
        destination = Path(dest)
        verified = destination / _VERIFIED_FILE
        if not verified.is_file():
            raise ArtifactNotFoundError(
                f"no verified dataset at {destination}; run `stc data fetch`"
            )
        recorded = verified.read_text(encoding="utf-8").strip()
        archive = destination.parent / f"{destination.name}.tar.gz"
        if not archive.is_file():
            raise IntegrityError(f"cached archive is missing: {archive.name}")
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            while chunk := handle.read(_CHUNK_SIZE):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != recorded or actual != self.sha256:
            raise ChecksumMismatchError(
                f"cached archive checksum mismatch: expected {self.sha256}, got {actual}"
            )
        return actual

    def _ref(self, destination: Path, *, cached: bool) -> RawDatasetRef:
        return RawDatasetRef(
            root=destination,
            sha256=self.sha256,
            size_bytes=self.size,
            source_url=self.url,
            cached=cached,
        )

    def _is_cached(self, destination: Path) -> bool:
        verified = destination / _VERIFIED_FILE
        if not verified.is_file():
            return False
        try:
            recorded = verified.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if recorded != self.sha256:
            return False
        return all((destination / name).is_file() for name in ALLOWED_MEMBERS.values())

    def _download(self, part: Path) -> None:
        digest = hashlib.sha256()
        downloaded = 0
        try:
            # The URL is the pinned S1 source (file:// is used by tests only).
            with (
                urllib.request.urlopen(  # noqa: S310
                    self.url, timeout=self.timeout
                ) as response,
                part.open("wb") as handle,
            ):
                while chunk := response.read(_CHUNK_SIZE):
                    downloaded += len(chunk)
                    digest.update(chunk)
                    handle.write(chunk)
        except (urllib.error.URLError, OSError) as error:
            part.unlink(missing_ok=True)
            raise DomainError(f"could not download the dataset from {self.url}: {error}") from error

        if downloaded != self.size:
            part.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"size mismatch for {self.url}: expected {self.size} bytes, got {downloaded}"
            )
        actual = digest.hexdigest()
        if actual != self.sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"sha256 mismatch for {self.url}: expected {self.sha256}, got {actual}"
            )

    def _extract(self, archive: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".massive-extract-", dir=destination.parent))
        try:
            found: set[str] = set()
            with tarfile.open(archive, "r:*") as tar:
                for member in tar:
                    self._validate_member(member)
                    if member.name not in ALLOWED_MEMBERS:
                        continue
                    if not member.isfile():
                        raise IntegrityError(
                            f"archive member {member.name!r} is not a regular file"
                        )
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise IntegrityError(f"archive member {member.name!r} cannot be read")
                    target = staging / ALLOWED_MEMBERS[member.name]
                    target.write_bytes(extracted.read())
                    found.add(member.name)

            missing = sorted(set(ALLOWED_MEMBERS) - found)
            if missing:
                raise IntegrityError(f"archive is missing required members: {missing}")
            for target_name in ALLOWED_MEMBERS.values():
                (staging / target_name).replace(destination / target_name)
            (destination / _VERIFIED_FILE).write_text(self.sha256 + "\n", encoding="utf-8")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _validate_member(member: tarfile.TarInfo) -> None:
        if member.issym() or member.islnk():
            raise IntegrityError(f"archive member {member.name!r} is a link")
        name = member.name
        path = PurePosixPath(name)
        if name.startswith("/") or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise IntegrityError(f"archive member {name!r} escapes the destination directory")
