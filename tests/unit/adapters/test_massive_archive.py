"""Tests for the pinned MASSIVE archive source."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from stc.adapters.data.massive_archive import MassiveArchiveSource
from stc.domain.errors import (
    ArtifactNotFoundError,
    ChecksumMismatchError,
    IntegrityError,
    OfflineError,
)

pytestmark = pytest.mark.unit

ALLOWED = {
    "1.0/data/es-ES.jsonl": b'{"id":"1"}\n',
    "1.0/LICENSE": b"CC BY 4.0\n",
    "1.0/NOTICE.md": b"Amazon MASSIVE\n",
}


def build_archive(
    path: Path,
    members: dict[str, bytes] | None = None,
    *,
    extra: list[tarfile.TarInfo] | None = None,
) -> tuple[str, int]:
    payload_members = ALLOWED if members is None else members
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in payload_members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for info in extra or []:
            archive.addfile(info, io.BytesIO(b"") if info.isfile() else None)
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def make_source(archive: Path, sha256: str, size: int, **kwargs: object) -> MassiveArchiveSource:
    return MassiveArchiveSource(
        url=archive.as_uri(),
        sha256=sha256,
        size=size,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, str, int]:
    path = tmp_path / "massive.tar.gz"
    sha256, size = build_archive(path)
    return path, sha256, size


def test_fetch_extracts_exactly_the_allowed_members(
    tmp_path: Path, archive: tuple[Path, str, int]
) -> None:
    path, sha256, size = archive
    dest = tmp_path / "data" / "raw" / "massive-1.0"

    ref = make_source(path, sha256, size).fetch(dest)

    assert ref.cached is False
    assert ref.root == dest
    assert ref.sha256 == sha256
    assert ref.size_bytes == size
    assert (dest / "es-ES.jsonl").read_bytes() == ALLOWED["1.0/data/es-ES.jsonl"]
    assert (dest / "LICENSE").read_bytes() == ALLOWED["1.0/LICENSE"]
    assert (dest / "NOTICE.md").read_bytes() == ALLOWED["1.0/NOTICE.md"]
    assert (dest / "VERIFIED").read_text(encoding="utf-8").strip() == sha256
    assert list(dest.parent.glob("*.part")) == []


def test_second_fetch_is_a_cache_hit_without_network(
    tmp_path: Path, archive: tuple[Path, str, int]
) -> None:
    path, sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"
    make_source(path, sha256, size).fetch(dest)
    path.unlink()

    ref = make_source(path, sha256, size).fetch(dest)

    assert ref.cached is True
    assert ref.root == dest
    assert (dest / "es-ES.jsonl").exists()


def test_force_redownloads_even_when_cached(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path, sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"
    make_source(path, sha256, size).fetch(dest)
    changed = tmp_path / "changed.tar.gz"
    changed_sha, changed_size = build_archive(
        changed, {**ALLOWED, "1.0/data/es-ES.jsonl": b'{"id":"2"}\n'}
    )

    ref = make_source(changed, changed_sha, changed_size).fetch(dest, force=True)

    assert ref.cached is False
    assert ref.sha256 == changed_sha
    assert (dest / "es-ES.jsonl").read_bytes() == b'{"id":"2"}\n'


def test_wrong_checksum_extracts_nothing_and_removes_the_part_file(
    tmp_path: Path, archive: tuple[Path, str, int]
) -> None:
    path, _sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"

    with pytest.raises(ChecksumMismatchError):
        make_source(path, "0" * 64, size).fetch(dest)

    assert not (dest / "es-ES.jsonl").exists()
    assert not (dest / "VERIFIED").exists()
    assert list((tmp_path / "raw").glob("*.part")) == []


def test_wrong_size_is_checksum_mismatch(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path, sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"

    with pytest.raises(ChecksumMismatchError):
        make_source(path, sha256, size + 1).fetch(dest)

    assert not (dest / "VERIFIED").exists()


def test_offline_refuses_to_download(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path, sha256, size = archive
    source = make_source(path, sha256, size, offline=True)

    with pytest.raises(OfflineError):
        source.fetch(tmp_path / "raw" / "massive-1.0")


def test_cache_hit_works_offline(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path, sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"
    make_source(path, sha256, size).fetch(dest)

    ref = make_source(path, sha256, size, offline=True).fetch(dest)

    assert ref.cached is True


def test_unrelated_members_are_ignored_not_extracted(
    tmp_path: Path, archive: tuple[Path, str, int]
) -> None:
    path = tmp_path / "many.tar.gz"
    sha256, size = build_archive(
        path,
        {
            **ALLOWED,
            "1.0/data/fr-FR.jsonl": b"bonjour\n",
            "1.0/README.md": b"readme\n",
        },
    )
    dest = tmp_path / "raw" / "massive-1.0"

    make_source(path, sha256, size).fetch(dest)

    assert sorted(child.name for child in dest.iterdir()) == [
        "LICENSE",
        "NOTICE.md",
        "VERIFIED",
        "es-ES.jsonl",
    ]


@pytest.mark.parametrize("name", ["../escape.txt", "../../escape.txt", "/absolute/escape.txt"])
def test_path_traversal_members_are_rejected(
    tmp_path: Path, archive: tuple[Path, str, int], name: str
) -> None:
    path = tmp_path / "evil.tar.gz"
    evil = tarfile.TarInfo(name=name)
    evil.size = 0
    sha256, size = build_archive(path, extra=[evil])
    dest = tmp_path / "raw" / "massive-1.0"

    with pytest.raises(IntegrityError):
        make_source(path, sha256, size).fetch(dest)

    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / "es-ES.jsonl").exists()


def test_symlink_members_are_rejected(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path = tmp_path / "symlink.tar.gz"
    link = tarfile.TarInfo(name="1.0/data/es-ES.jsonl")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    sha256, size = build_archive(path, extra=[link])
    dest = tmp_path / "raw" / "massive-1.0"

    with pytest.raises(IntegrityError):
        make_source(path, sha256, size).fetch(dest)

    assert not (dest / "es-ES.jsonl").is_symlink()


def test_verify_rehashes_the_cached_archive(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path, sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"
    source = make_source(path, sha256, size)
    source.fetch(dest)

    assert source.verify(dest) == sha256


def test_verify_detects_a_tampered_archive(tmp_path: Path, archive: tuple[Path, str, int]) -> None:
    path, sha256, size = archive
    dest = tmp_path / "raw" / "massive-1.0"
    source = make_source(path, sha256, size)
    source.fetch(dest)
    cached = tmp_path / "raw" / "massive-1.0.tar.gz"
    cached.write_bytes(cached.read_bytes() + b"tampered")

    with pytest.raises(ChecksumMismatchError):
        source.verify(dest)


def test_verify_without_cache_raises_artifact_not_found(tmp_path: Path) -> None:
    source = MassiveArchiveSource()

    with pytest.raises(ArtifactNotFoundError):
        source.verify(tmp_path / "raw" / "massive-1.0")


def test_missing_required_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partial.tar.gz"
    sha256, size = build_archive(path, {"1.0/data/es-ES.jsonl": b"{}\n"})
    dest = tmp_path / "raw" / "massive-1.0"

    with pytest.raises(IntegrityError):
        make_source(path, sha256, size).fetch(dest)

    assert not (dest / "VERIFIED").exists()
