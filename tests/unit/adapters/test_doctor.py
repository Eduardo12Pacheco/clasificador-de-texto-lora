"""Environment reports expose the T7/T11 diagnostic fields."""

from __future__ import annotations

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
