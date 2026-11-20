"""CLI tests for the ``stc data`` command group."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stc import bootstrap
from stc.adapters.cli.app import app
from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.adapters.data.massive_archive import MassiveArchiveSource
from stc.adapters.data.massive_parser import MassiveParser
from stc.application.use_cases.prepare_data import PrepareDataPipeline
from stc.config import Settings
from stc.domain.splits import Split

pytestmark = pytest.mark.unit

RAW_ROWS = [
    ("1", "train", "play_music", "play", "Pon música", "w1"),
    ("2", "train", "alarm_set", "alarm", "Pon una alarma", "w2"),
    ("100", "dev", "alarm_set", "alarm", "otra alarma", "w3"),
    ("200", "test", "play_music", "play", "otra canción", "w3"),
]


def make_tarball(tmp_path: Path) -> tuple[Path, str, int]:
    lines = [
        json.dumps(
            {
                "id": row_id,
                "locale": "es-ES",
                "partition": partition,
                "scenario": scenario,
                "intent": intent,
                "utt": text,
                "worker_id": worker,
            },
            ensure_ascii=False,
        )
        + "\n"
        for row_id, partition, intent, scenario, text, worker in RAW_ROWS
    ]
    path = tmp_path / "massive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        content = "".join(lines).encode()
        info = tarfile.TarInfo(name="1.0/data/es-ES.jsonl")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
        for name in ("1.0/LICENSE", "1.0/NOTICE.md"):
            payload = b"notice\n"
            extra = tarfile.TarInfo(name=name)
            extra.size = len(payload)
            archive.addfile(extra, io.BytesIO(payload))
    data = path.read_bytes()
    return path, hashlib.sha256(data).hexdigest(), len(data)


class FakeFixtures:
    def load(self, split: Split) -> tuple[object, ...]:
        return tuple(range({"train": 6, "dev": 2, "test": 2}[split.value]))

    def source_size_bytes(self) -> int:
        return 1234


def build_services(tmp_path: Path, *, offline: bool = False) -> bootstrap.DataServices:
    path, sha256, size = make_tarball(tmp_path)
    settings = Settings(home=tmp_path, offline=offline)
    raw_dir = tmp_path / "raw" / "massive-1.0"
    processed_dir = tmp_path / "processed" / "massive-1.0"
    source = MassiveArchiveSource(url=path.as_uri(), sha256=sha256, size=size, offline=offline)
    repository = JsonlDatasetRepository(processed_dir)
    pipeline = PrepareDataPipeline(
        source=source,
        parser=MassiveParser(),
        repository=repository,
        raw_dir=raw_dir,
        stc_version="0.1.0",
    )
    return bootstrap.DataServices(
        settings=settings,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        source=source,
        repository=repository,
        pipeline=pipeline,
        fixtures=FakeFixtures(),  # type: ignore[arg-type]
    )


@pytest.fixture
def services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bootstrap.DataServices:
    built = build_services(tmp_path)
    monkeypatch.setattr(bootstrap, "build_data_services", lambda **_: built)
    return built


def test_fetch_prints_checksum_then_cache_hit(services: bootstrap.DataServices) -> None:
    runner = CliRunner()

    first = runner.invoke(app, ["data", "fetch"])
    second = runner.invoke(app, ["data", "fetch"])

    assert first.exit_code == 0, first.output
    assert services.source.sha256 in first.stdout  # type: ignore[attr-defined]
    assert "sha256 OK" in first.stdout
    assert second.exit_code == 0
    assert "cache hit" in second.stdout
    assert (services.raw_dir / "VERIFIED").is_file()


def test_fetch_reports_offline_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_services(tmp_path, offline=True)
    monkeypatch.setattr(bootstrap, "build_data_services", lambda **_: built)

    result = CliRunner().invoke(app, ["data", "fetch"])

    assert result.exit_code == 5
    assert "offline" in result.stderr.lower()


def test_verify_after_fetch_then_detects_tampering(services: bootstrap.DataServices) -> None:
    runner = CliRunner()
    runner.invoke(app, ["data", "fetch"])
    ok = runner.invoke(app, ["data", "verify"])
    assert ok.exit_code == 0, ok.output
    assert "verified" in ok.stdout

    archive = services.raw_dir.parent / "massive-1.0.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"x")
    tampered = runner.invoke(app, ["data", "verify"])

    assert tampered.exit_code == 3
    assert "checksum" in tampered.stderr.lower()


def test_verify_without_cache_exits_4(services: bootstrap.DataServices) -> None:
    result = CliRunner().invoke(app, ["data", "verify"])

    assert result.exit_code == 4
    assert "stc data fetch" in result.stderr


def test_prepare_writes_splits(services: bootstrap.DataServices) -> None:
    result = CliRunner().invoke(app, ["data", "prepare"])

    assert result.exit_code == 0, result.output
    assert "train=2" in result.stdout
    assert "dev=1" in result.stdout
    assert "test=1" in result.stdout
    repository = JsonlDatasetRepository(services.processed_dir)
    assert len(repository.load(Split.TRAIN)) == 2
    assert repository.meta().processed_counts == {"train": 2, "dev": 1, "test": 1}


def test_audit_writes_reports(services: bootstrap.DataServices, tmp_path: Path) -> None:
    out = tmp_path / "reports"

    result = CliRunner().invoke(app, ["data", "audit", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "data_audit.json").is_file()
    assert (out / "data_audit.md").is_file()
    payload = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    assert payload["raw_counts"] == {"train": 2, "dev": 1, "test": 1}
    assert "data_audit.json" in result.stdout


def test_fixture_info_reports_counts(services: bootstrap.DataServices) -> None:
    result = CliRunner().invoke(app, ["data", "fixture", "info"])

    assert result.exit_code == 0, result.output
    assert "train 6" in result.stdout
    assert "dev 2" in result.stdout
    assert "test 2" in result.stdout
    assert "1234 bytes" in result.stdout


def test_data_help_lists_every_command() -> None:
    result = CliRunner().invoke(app, ["data", "--help"])

    assert result.exit_code == 0
    for command in ("fetch", "verify", "prepare", "audit", "fixture"):
        assert command in result.stdout
