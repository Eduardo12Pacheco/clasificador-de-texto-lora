"""Bundle manifests are the trust anchor: every hash and every extra file matters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stc.adapters.inference.bundle_io import (
    load_calibration,
    load_labels,
    model_file,
    read_manifest,
    refresh_manifest,
    sha256_file,
    unpack_zip,
    verify_bundle,
)
from stc.domain.errors import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Return a writable copy of the committed tiny bundle."""
    target = tmp_path / "bundle"
    shutil.copytree(TINY_BUNDLE, target)
    return target


def test_verify_bundle_accepts_the_committed_fixture() -> None:
    manifest = verify_bundle(TINY_BUNDLE)
    assert manifest["kind"] == "onnx-seqcls"
    assert manifest["name"] == "stc-tiny-test"
    assert len(manifest["files"]) == 4


def test_verify_bundle_rejects_a_flipped_byte(bundle: Path) -> None:
    data = bytearray((bundle / "model.onnx").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (bundle / "model.onnx").write_bytes(bytes(data))
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        verify_bundle(bundle)


def test_verify_bundle_reports_a_missing_file(bundle: Path) -> None:
    (bundle / "labels.json").unlink()
    with pytest.raises(IntegrityError, match="missing file"):
        verify_bundle(bundle)


def test_verify_bundle_reports_an_unexpected_file(bundle: Path) -> None:
    (bundle / "rogue.onnx").write_bytes(b"not a real model")
    with pytest.raises(IntegrityError, match="unexpected file"):
        verify_bundle(bundle)


def test_optional_companions_may_appear_before_a_repack(bundle: Path) -> None:
    (bundle / "NOTICE").write_text("notice\n", encoding="utf-8")
    verify_bundle(bundle)


def test_refresh_manifest_hashes_new_companions(bundle: Path) -> None:
    (bundle / "NOTICE").write_text("notice\n", encoding="utf-8")
    manifest = refresh_manifest(bundle)
    paths = {entry["path"] for entry in manifest["files"]}
    assert "NOTICE" in paths
    verify_bundle(bundle)
    assert (bundle / "manifest.json").read_text(encoding="utf-8").endswith("\n")


def test_refresh_manifest_hashes_a_nested_baseline_manifest(bundle: Path) -> None:
    baseline = bundle / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text('{"kind": "tfidf-linear"}', encoding="utf-8")
    (baseline / "baseline.skops").write_bytes(b"fake baseline")

    manifest = refresh_manifest(bundle)

    paths = {entry["path"] for entry in manifest["files"]}
    assert {"baseline/manifest.json", "baseline/baseline.skops"} <= paths
    verify_bundle(bundle)


def test_read_manifest_rejects_an_unsupported_version(bundle: Path) -> None:
    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload["bundle_version"] = 99
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntegrityError, match="unsupported bundle_version"):
        read_manifest(bundle)


def test_read_manifest_rejects_an_unsafe_entry_path(bundle: Path) -> None:
    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../escape"
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntegrityError, match="unsafe bundle path"):
        verify_bundle(bundle)


def test_loaders_expose_labels_calibration_and_model_file() -> None:
    labels = load_labels(TINY_BUNDLE)
    assert labels.n_intents == 60
    assert labels.n_scenarios == 18
    calibration = load_calibration(TINY_BUNDLE)
    assert calibration is not None
    assert calibration.temperature == pytest.approx(1.5)
    assert calibration.review_threshold == pytest.approx(0.5)
    manifest = read_manifest(TINY_BUNDLE)
    assert model_file(TINY_BUNDLE, manifest).name == "model.onnx"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "payload.bin"
    path.write_bytes(b"spanish-text-classifier-lora")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unpack_zip_rejects_zip_slip(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "boom")
    with pytest.raises(IntegrityError, match="unsafe bundle path"):
        unpack_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_unpack_zip_extracts_a_bundle(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in ("manifest.json", "labels.json"):
            handle.write(TINY_BUNDLE / name, arcname=name)
    root = unpack_zip(archive, tmp_path / "out")
    assert (root / "manifest.json").is_file()
    # The extracted tree is internally inconsistent (the manifest lists more
    # files), which the verifier must catch rather than silently accept.
    with pytest.raises(IntegrityError, match="missing file"):
        verify_bundle(root)
