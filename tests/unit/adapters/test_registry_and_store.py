"""Registry metadata validation and the filesystem model store."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from stc.adapters.hub.registry import ModelRegistry, RegistryEntry
from stc.adapters.inference.bundle_io import BundleArchiveAdapter
from stc.adapters.storage.fs_model_store import FsModelStore
from stc.domain.errors import ArtifactNotFoundError, IntegrityError, InvalidInputError

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
REGISTRY = REPO_ROOT / "src" / "stc" / "model_registry.json"


def _entry(**overrides: object) -> RegistryEntry:
    values: dict[str, object] = {
        "name": "stc-es-intent",
        "version": "0.1.0",
        "url": "https://example.invalid/model.zip",
        "sha256": "a" * 64,
        "size_bytes": 1234,
        "license": "Apache-2.0",
    }
    values.update(overrides)
    return RegistryEntry(**values)  # type: ignore[arg-type]


def test_committed_registry_is_valid_and_empty() -> None:
    registry = ModelRegistry.from_file(REGISTRY)
    assert registry.entries == ()


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = ModelRegistry.empty().upsert(_entry()).upsert(_entry(version="0.2.0"))
    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = ModelRegistry.from_file(path)
    assert loaded.find("stc-es-intent").version == "0.1.0"
    assert loaded.find("stc-es-intent", "0.2.0").version == "0.2.0"
    with pytest.raises(ArtifactNotFoundError):
        loaded.find("nope")


def test_upsert_replaces_the_same_version(tmp_path: Path) -> None:
    registry = ModelRegistry.empty().upsert(_entry()).upsert(_entry(sha256="b" * 64))
    assert len(registry.entries) == 1
    assert registry.entries[0].sha256 == "b" * 64


def test_registry_rejects_bad_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"registry_version": 2, "models": []}', encoding="utf-8")
    with pytest.raises(IntegrityError):
        ModelRegistry.from_file(bad)
    with pytest.raises(IntegrityError):
        _entry(sha256="not-a-hash")
    with pytest.raises(IntegrityError):
        _entry(size_bytes=0)


def test_store_installs_and_lists_a_bundle(tmp_path: Path) -> None:
    store = FsModelStore(tmp_path / "models", bundle=BundleArchiveAdapter())
    installed = store.save(TINY_BUNDLE, "stc-tiny-test", "0.0.1")
    assert installed == tmp_path / "models" / "stc-tiny-test" / "0.0.1"
    assert store.load("stc-tiny-test", "0.0.1") == installed
    listed = store.list_installed()
    assert len(listed) == 1
    assert listed[0].model_id == "stc-tiny-test"
    assert listed[0].version == "0.0.1"


def test_store_rejects_path_components(tmp_path: Path) -> None:
    store = FsModelStore(tmp_path / "models", bundle=BundleArchiveAdapter())
    with pytest.raises(InvalidInputError):
        store.save(TINY_BUNDLE, "../escape", "0.0.1")
    with pytest.raises(ArtifactNotFoundError):
        store.load("missing", "0.0.1")


def test_store_refuses_a_tampered_bundle(tmp_path: Path) -> None:
    copy = tmp_path / "bundle"
    shutil.copytree(TINY_BUNDLE, copy)
    data = bytearray((copy / "labels.json").read_bytes())
    data[0] ^= 0xFF
    (copy / "labels.json").write_bytes(bytes(data))
    store = FsModelStore(tmp_path / "models", bundle=BundleArchiveAdapter())
    with pytest.raises(IntegrityError):
        store.save(copy, "stc-tiny-test", "0.0.1")


def test_store_lists_an_empty_root(tmp_path: Path) -> None:
    store = FsModelStore(tmp_path / "missing", bundle=BundleArchiveAdapter())
    assert store.list_installed() == ()
