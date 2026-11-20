"""CLI wiring for export-related commands on the committed tiny bundle."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from stc.adapters.cli.app import app
from stc.adapters.hub.registry import ModelRegistry
from stc.adapters.inference.bundle_io import verify_bundle

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _invoke(args: list[str], home: Path, **kwargs: object):
    env = {"STC_HOME": str(home)}
    return runner.invoke(app, args, env=env, **kwargs)


def test_model_verify_accepts_the_fixture(tmp_path: Path) -> None:
    result = _invoke(["model", "verify", str(TINY_BUNDLE)], tmp_path)
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_model_verify_rejects_a_tampered_bundle(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    shutil.copytree(TINY_BUNDLE, target)
    data = bytearray((target / "labels.json").read_bytes())
    data[0] ^= 0xFF
    (target / "labels.json").write_bytes(bytes(data))
    result = _invoke(["model", "verify", str(target)], tmp_path)
    assert result.exit_code == 3
    assert "checksum mismatch" in result.output


def test_bundle_pack_is_deterministic_and_updates_the_registry(tmp_path: Path) -> None:
    first = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "a.zip")],
        tmp_path,
    )
    assert first.exit_code == 0, first.output
    second = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "b.zip")],
        tmp_path,
    )
    assert second.exit_code == 0, second.output
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    assert first.stdout.splitlines()[0] == second.stdout.splitlines()[0]
    data = (tmp_path / "a.zip").read_bytes()
    assert hashlib.sha256(data).hexdigest() in first.stdout

    registry = tmp_path / "registry.json"
    third = _invoke(
        [
            "bundle",
            "pack",
            "--bundle",
            str(TINY_BUNDLE),
            "--out",
            str(tmp_path / "c.zip"),
            "--update-registry",
            str(registry),
            "--url",
            "https://example.invalid/stc-tiny-test-0.0.1.zip",
        ],
        tmp_path,
    )
    assert third.exit_code == 0, third.output
    entry = ModelRegistry.from_file(registry).find("stc-tiny-test")
    assert entry.sha256 == hashlib.sha256((tmp_path / "c.zip").read_bytes()).hexdigest()
    assert entry.size_bytes == (tmp_path / "c.zip").stat().st_size


def test_bundle_pack_requires_url_with_registry(tmp_path: Path) -> None:
    result = _invoke(
        [
            "bundle",
            "pack",
            "--bundle",
            str(TINY_BUNDLE),
            "--out",
            str(tmp_path / "a.zip"),
            "--update-registry",
            str(tmp_path / "registry.json"),
        ],
        tmp_path,
    )
    assert result.exit_code == 2


def test_bundle_verify_accepts_a_packed_archive(tmp_path: Path) -> None:
    result = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "a.zip")],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    extract = tmp_path / "extracted"
    with zipfile.ZipFile(tmp_path / "a.zip") as handle:
        handle.extractall(extract)
    verify = _invoke(["bundle", "verify", str(extract)], tmp_path)
    assert verify.exit_code == 0, verify.output


def test_pull_from_file_then_list_info_and_verify(tmp_path: Path) -> None:
    pack = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "a.zip")],
        tmp_path,
    )
    assert pack.exit_code == 0, pack.output
    home = tmp_path / "home"
    pull = _invoke(["model", "pull", "--from-file", str(tmp_path / "a.zip")], home)
    assert pull.exit_code == 0, pull.output
    assert "stc-tiny-test 0.0.1" in pull.output

    listing = _invoke(["model", "list"], home)
    assert listing.exit_code == 0, listing.output
    assert "stc-tiny-test" in listing.output

    info = _invoke(["model", "info"], home)
    assert info.exit_code == 0, info.output
    assert "kind=onnx-seqcls" in info.output

    verify = _invoke(["model", "verify", "stc-tiny-test"], home)
    assert verify.exit_code == 0, verify.output


def test_offline_pull_without_a_file_exits_5(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["model", "pull"],
        env={"STC_HOME": str(tmp_path), "STC_OFFLINE": "1"},
    )
    assert result.exit_code == 5
    assert "STC_OFFLINE" in result.output


def test_pull_from_file_works_offline(tmp_path: Path) -> None:
    pack = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "a.zip")],
        tmp_path,
    )
    assert pack.exit_code == 0, pack.output
    result = runner.invoke(
        app,
        ["model", "pull", "--from-file", str(tmp_path / "a.zip")],
        env={"STC_HOME": str(tmp_path / "home"), "STC_OFFLINE": "1"},
    )
    assert result.exit_code == 0, result.output


def test_calibrate_writes_calibration_and_refreshes_the_manifest(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    shutil.copytree(TINY_BUNDLE, target)
    result = _invoke(
        [
            "calibrate",
            "--bundle",
            str(target),
            "--dataset",
            "fixture",
            "--reports",
            str(tmp_path / "reports"),
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    calibration = json.loads((target / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["temperature"] > 0
    assert 0.0 <= calibration["review_threshold"] <= 1.0
    assert (tmp_path / "reports" / "calibration_dev.json").is_file()
    verify = _invoke(["model", "verify", str(target)], tmp_path)
    assert verify.exit_code == 0, verify.output


def test_benchmark_writes_the_report(tmp_path: Path) -> None:
    out = tmp_path / "benchmark.json"
    result = _invoke(
        [
            "benchmark",
            "--bundle",
            str(TINY_BUNDLE),
            "--dataset",
            "fixture",
            "--threads",
            "1",
            "--runs",
            "3",
            "--out",
            str(out),
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["threads"][0]["threads"] == 1
    assert payload["threads"][0]["p50_ms"] >= 0
    assert payload["onnxruntime_version"] != "unknown"


def test_bundle_verify_reports_a_missing_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _invoke(["bundle", "verify", str(empty)], tmp_path)
    assert result.exit_code == 3


def test_help_lists_every_new_command(tmp_path: Path) -> None:
    result = _invoke(["--help"], tmp_path)
    assert result.exit_code == 0
    for command in ("export", "calibrate", "benchmark", "bundle", "model"):
        assert command in result.output


def test_export_help_exits_zero(tmp_path: Path) -> None:
    result = _invoke(["export", "--help"], tmp_path)
    assert result.exit_code == 0
    assert "--quantize" in _plain(result.output)


def test_packed_bundle_carries_the_legal_assets(tmp_path: Path) -> None:
    pack = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "a.zip")],
        tmp_path,
    )
    assert pack.exit_code == 0, pack.output
    with zipfile.ZipFile(tmp_path / "a.zip") as handle:
        names = set(handle.namelist())
        notice = handle.read("NOTICE").decode("utf-8")
    assert {"LICENSE", "NOTICE", "manifest.json"} <= names
    assert "Amazon MASSIVE" in notice
    assert "CC BY 4.0" in notice


def test_model_info_of_a_missing_model_exits_4(tmp_path: Path) -> None:
    result = _invoke(["model", "info", "nope"], tmp_path)
    assert result.exit_code == 4


def test_verify_bundle_helper_accepts_the_fixture() -> None:
    assert verify_bundle(TINY_BUNDLE)["name"] == "stc-tiny-test"


def test_cli_captured_output_is_utf8(tmp_path: Path) -> None:
    result = _invoke(["model", "verify", str(TINY_BUNDLE)], tmp_path)
    assert isinstance(result.stdout, str)
    assert io.StringIO(result.stdout).readline().startswith("OK")


def test_pack_adapter_cli(tmp_path: Path) -> None:
    run = tmp_path / "seed-13"
    run.mkdir()
    (run / "adapter").mkdir()
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "run.json").write_text('{"schema_version": 1}', encoding="utf-8")
    result = _invoke(
        ["bundle", "pack-adapter", "--run", str(run), "--out", str(tmp_path / "adapter.zip")],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "sha256" in result.output
    with zipfile.ZipFile(tmp_path / "adapter.zip") as handle:
        assert "adapter/adapter_config.json" in handle.namelist()


def test_pull_from_a_registry_entry(tmp_path: Path) -> None:
    packed = _invoke(
        ["bundle", "pack", "--bundle", str(TINY_BUNDLE), "--out", str(tmp_path / "a.zip")],
        tmp_path,
    )
    assert packed.exit_code == 0, packed.output
    digest = hashlib.sha256((tmp_path / "a.zip").read_bytes()).hexdigest()
    registry = {
        "registry_version": 1,
        "models": [
            {
                "name": "stc-tiny-test",
                "version": "0.0.1",
                "url": (tmp_path / "a.zip").resolve().as_uri(),
                "sha256": digest,
                "size_bytes": (tmp_path / "a.zip").stat().st_size,
                "license": "Apache-2.0",
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    home = tmp_path / "home"
    result = _invoke(
        [
            "model",
            "pull",
            "stc-tiny-test",
            "--registry",
            str(registry_path),
        ],
        home,
    )
    assert result.exit_code == 0, result.output
    assert "stc-tiny-test 0.0.1" in result.output
