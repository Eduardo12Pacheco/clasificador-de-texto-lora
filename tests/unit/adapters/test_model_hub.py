"""Model hub: verified downloads, file:// support, offline refusal and packing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stc.adapters.hub.github_release import GithubReleaseHub, download_verified
from stc.adapters.inference.bundle_io import (
    BundleArchiveAdapter,
    sha256_file,
    unpack_zip,
    verify_bundle,
    write_bundle_zip,
)
from stc.adapters.storage.fs_model_store import FsModelStore
from stc.application.use_cases.pack_bundle import PackBundle
from stc.application.use_cases.pull_model import PullModel
from stc.domain.errors import IntegrityError, OfflineError

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
ARCHIVE = BundleArchiveAdapter()


@pytest.fixture
def tiny_zip(tmp_path: Path) -> Path:
    return write_bundle_zip(TINY_BUNDLE, tmp_path / "tiny.zip")


@pytest.fixture
def store(tmp_path: Path) -> FsModelStore:
    return FsModelStore(tmp_path / "models", bundle=ARCHIVE)


def _entry(path: Path, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": "stc-tiny-test",
        "version": "0.0.1",
        "url": path.resolve().as_uri(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    entry.update(overrides)
    return entry


def test_download_verified_file_url(tmp_path: Path, tiny_zip: Path) -> None:
    target = tmp_path / "download.zip"
    download_verified(tiny_zip.as_uri(), target, sha256_file(tiny_zip))
    assert target.read_bytes() == tiny_zip.read_bytes()


def test_download_rejects_a_wrong_checksum(tmp_path: Path, tiny_zip: Path) -> None:
    target = tmp_path / "download.zip"
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        download_verified(tiny_zip.as_uri(), target, "0" * 64)
    assert not target.exists()


def test_hub_pulls_and_extracts(tmp_path: Path, tiny_zip: Path) -> None:
    hub = GithubReleaseHub(ARCHIVE)
    extracted = hub.pull(_entry(tiny_zip), tmp_path / "pull")
    assert (extracted / "model.onnx").is_file()
    verify_bundle(extracted)


def test_hub_is_offline(tmp_path: Path, tiny_zip: Path) -> None:
    hub = GithubReleaseHub(ARCHIVE, offline=True)
    with pytest.raises(OfflineError):
        hub.pull(_entry(tiny_zip), tmp_path / "pull")


def test_pull_from_file_installs_a_bundle(tmp_path: Path, store: FsModelStore) -> None:
    archive = write_bundle_zip(TINY_BUNDLE, tmp_path / "a.zip")
    pull = PullModel(GithubReleaseHub(ARCHIVE), store, ARCHIVE)
    result = pull.run_from_file(archive, tmp_path / "scratch")
    assert result.name == "stc-tiny-test"
    assert result.version == "0.0.1"
    assert result.path == store.load("stc-tiny-test", "0.0.1")


def test_pull_from_entry_installs_a_bundle(
    tmp_path: Path, tiny_zip: Path, store: FsModelStore
) -> None:
    pull = PullModel(GithubReleaseHub(ARCHIVE), store, ARCHIVE)
    result = pull.run_entry(_entry(tiny_zip), tmp_path / "scratch")
    assert store.load("stc-tiny-test", "0.0.1") == result.path


def test_pull_from_a_corrupt_zip_fails(tmp_path: Path, store: FsModelStore) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    pull = PullModel(GithubReleaseHub(ARCHIVE), store, ARCHIVE)
    with pytest.raises(IntegrityError):
        pull.run_from_file(archive, tmp_path / "scratch")


def test_repacked_zip_is_extractable(tmp_path: Path, tiny_zip: Path) -> None:
    root = unpack_zip(tiny_zip, tmp_path / "unpacked")
    assert verify_bundle(root)["name"] == "stc-tiny-test"


def test_pack_is_deterministic(tmp_path: Path) -> None:
    packer = PackBundle(ARCHIVE)
    first = packer.run(
        TINY_BUNDLE,
        tmp_path / "a.zip",
        license_text="Apache License\n",
        notice_text="NOTICE\n",
    )
    second = packer.run(
        TINY_BUNDLE,
        tmp_path / "b.zip",
        license_text="Apache License\n",
        notice_text="NOTICE\n",
    )
    assert first.sha256 == second.sha256
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()


def test_pack_assembles_the_release_layout(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text('{"kind": "tfidf-linear"}', encoding="utf-8")
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# card\n", encoding="utf-8")
    packer = PackBundle(ARCHIVE)
    result = packer.run(
        TINY_BUNDLE,
        tmp_path / "release.zip",
        baseline_dir=baseline,
        model_card=card,
        license_text="Apache License\n",
        notice_text="NOTICE\n",
    )
    root = unpack_zip(result.path, tmp_path / "out")
    assert (root / "LICENSE").is_file()
    assert (root / "NOTICE").is_file()
    assert (root / "MODEL_CARD.md").is_file()
    assert (root / "baseline" / "manifest.json").is_file()
    manifest = verify_bundle(root)
    paths = {entry["path"] for entry in manifest["files"]}
    assert {"LICENSE", "NOTICE", "MODEL_CARD.md", "baseline/manifest.json"} <= paths
    assert result.name == "stc-tiny-test"
    assert result.size_bytes == result.path.stat().st_size


def test_pack_adapter_bundles_a_run(tmp_path: Path) -> None:
    run = tmp_path / "seed-13"
    run.mkdir()
    (run / "adapter").mkdir()
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    result = PackBundle(ARCHIVE).run_adapter(run, tmp_path / "adapter.zip", tmp_path / "scratch")
    assert result.size_bytes > 0
    import zipfile

    with zipfile.ZipFile(result.path) as handle:
        names = set(handle.namelist())
    assert "adapter/adapter_config.json" in names
    assert "run.json" in names


def test_pack_adapter_requires_an_adapter(tmp_path: Path) -> None:
    run = tmp_path / "seed-13"
    run.mkdir()
    with pytest.raises(FileNotFoundError):
        PackBundle(ARCHIVE).run_adapter(run, tmp_path / "adapter.zip", tmp_path / "scratch")


def test_pack_copies_a_baseline_directory(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    shutil.copytree(TINY_BUNDLE, baseline)
    result = PackBundle(ARCHIVE).run(
        TINY_BUNDLE,
        tmp_path / "release.zip",
        baseline_dir=baseline,
        license_text="license\n",
        notice_text="notice\n",
    )
    root = unpack_zip(result.path, tmp_path / "out")
    assert (root / "baseline" / "model.onnx").is_file()
