"""Download model bundles from a GitHub Release (or any pinned URL) with sha256."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from stc.domain.errors import IntegrityError, InvalidInputError, OfflineError

__all__ = ["GithubReleaseHub", "download_verified"]


class Unpacker(Protocol):
    """Extracts a verified bundle archive."""

    def unpack(self, archive: Path, dest: Path) -> Path:
        """Extract ``archive`` under ``dest`` and return the bundle root."""
        ...


_CHUNK = 1024 * 1024


def download_verified(url: str, dest: Path, expected_sha256: str, *, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``dest``, verifying the sha256 before returning."""
    digest = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The URL comes from the committed registry, never from user input.
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310
            dest.open("wb") as handle,
        ):
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                digest.update(chunk)
                handle.write(chunk)
    except (urllib.error.URLError, OSError) as error:
        dest.unlink(missing_ok=True)
        raise InvalidInputError(f"cannot download model bundle from {url}: {error}") from error
    actual = digest.hexdigest()
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise IntegrityError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    return dest


class GithubReleaseHub:
    """Fetch release assets over HTTPS/``file://`` and extract them safely."""

    def __init__(self, unpacker: Unpacker, *, offline: bool = False, timeout: float = 60.0) -> None:
        self._unpacker = unpacker
        self._offline = offline
        self._timeout = timeout

    def pull(self, entry: Mapping[str, object], dest: Path) -> Path:
        """Download the entry, verify its sha256 and return the extracted bundle."""
        if self._offline:
            raise OfflineError("network access is disabled by STC_OFFLINE")
        url = entry.get("url")
        expected = entry.get("sha256")
        if not isinstance(url, str) or not url:
            raise InvalidInputError("hub entry needs a url")
        if not isinstance(expected, str) or not expected:
            raise InvalidInputError("hub entry needs a sha256")
        target = Path(dest)
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as scratch:
            archive = Path(scratch) / "model.zip"
            download_verified(url, archive, expected, timeout=self._timeout)
            return self._unpacker.unpack(archive, target / "extracted")
