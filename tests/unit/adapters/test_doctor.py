"""Environment reports expose the T7/T11 diagnostic fields."""

from __future__ import annotations

import json

import pytest

from stc.adapters.cli.commands.doctor import build_environment_report


def test_basic_report_has_identity_and_library_versions() -> None:
    report = build_environment_report(gpu=False)

    assert report["schema_version"] == 1
    assert report["gpu_requested"] is False
    assert len(str(report["git_sha"])) >= 7
    versions = report["library_versions"]
    assert isinstance(versions, dict)
    assert "torch" in versions


@pytest.mark.torch
def test_gpu_report_has_cuda_fields() -> None:
    pytest.importorskip("torch")

    report = build_environment_report(gpu=True)

    assert report["gpu_requested"] is True
    for key in (
        "cuda_available",
        "device_name",
        "total_memory_mib",
        "bf16_supported",
        "driver_version",
    ):
        assert key in report
    assert isinstance(report["cuda_available"], bool)


def test_report_adds_onnxruntime_settings_paths_and_installed_models() -> None:
    report = build_environment_report(gpu=False)

    runtime = report["onnxruntime"]
    assert isinstance(runtime, dict)
    assert runtime["version"] != ""
    assert "CPUExecutionProvider" in runtime["providers"]
    settings = report["settings"]
    assert isinstance(settings, dict)
    assert "threads" in settings
    paths = report["paths"]
    assert isinstance(paths, dict)
    assert {"home", "data_dir", "models"} <= set(paths)
    assert isinstance(report["installed_models"], list)


def test_doctor_json_cli_exits_zero(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from stc.adapters.cli.app import app

    monkeypatch.setenv("STC_HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert "onnxruntime" in payload


def test_doctor_exits_one_when_onnxruntime_cannot_be_imported(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from stc.adapters.cli.app import app
    from stc.adapters.cli.commands import doctor as doctor_module

    def unavailable() -> dict[str, object]:
        raise ImportError("onnxruntime is missing")

    monkeypatch.setattr(doctor_module, "_onnxruntime_environment", unavailable)

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "onnxruntime" in result.stderr.lower()
