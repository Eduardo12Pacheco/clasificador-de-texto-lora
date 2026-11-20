"""The train CLI wires the smoke recipe end to end on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from typer.testing import CliRunner

import stc.adapters.training.trainer as trainer_module
from stc.adapters.cli.app import app

pytestmark = pytest.mark.torch


def test_train_lora_cli_writes_seed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer_module, "device_for_training", lambda: torch.device("cpu"))

    result = CliRunner().invoke(
        app,
        [
            "train",
            "lora",
            "--config",
            "configs/smoke.toml",
            "--dataset",
            "fixture",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dev macro_f1=" in result.output
    assert (tmp_path / "seed-13" / "run.json").is_file()
    assert (tmp_path / "selected.json").is_file()
