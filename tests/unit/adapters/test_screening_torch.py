"""Screening resumes finished runs and records the T11 contract fields."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stc.adapters.data.fixture_source import FixtureRepository
from stc.adapters.tracking.null_tracker import NullTracker
from stc.adapters.training.screening import ScreeningConfig, ScreeningRunner
from stc.domain.splits import Split

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


def _payload() -> dict[str, object]:
    return {
        "base": {
            "backbone": "stc-tiny-test",
            "method": "lora",
            "tokenizer_path": "tests/fixtures/tiny/tokenizer.json",
            "r": 4,
            "alpha": 8,
            "lora_dropout": 0.0,
            "epochs": 1,
            "batch_size": 16,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "class_weighting": "none",
        },
        "screening": {
            "backbones": ["stc-tiny-test"],
            "seeds": [13],
            "learning_rates": {"stc-tiny-test": 1e-2},
        },
    }


def test_screening_writes_runs_and_resumes(tmp_path: Path) -> None:
    repository = FixtureRepository()
    config = ScreeningConfig.from_mapping(_payload())
    runner = ScreeningRunner(
        list(repository.load(Split.TRAIN)),
        list(repository.load(Split.DEV)),
        tmp_path,
        config,
        tracker_factory=lambda run_dir: NullTracker(),
        device=CPU,
    )

    first = runner.run()

    assert len(first) == 1
    row = first[0]
    assert row.backbone == "stc-tiny-test"
    assert row.seed == 13
    assert row.exportable is False
    assert row.trainable_params > 0
    assert row.cpu_latency_ms_p50 >= 0.0
    assert (tmp_path / "stc-tiny-test-seed-13" / "run.json").is_file()

    second = runner.run()

    assert second[0].dev_macro_f1 == pytest.approx(row.dev_macro_f1)
    assert second[0].exportable is False
