"""Optuna search writes its contract files and resumes without new trials."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stc.adapters.data.fixture_source import FixtureRepository
from stc.adapters.tracking.null_tracker import NullTracker
from stc.adapters.training.search import OptunaSearcher
from stc.domain.splits import Split

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


def _payload() -> dict[str, object]:
    return {
        "base": {
            "backbone": "stc-tiny-test",
            "method": "lora",
            "tokenizer_path": "tests/fixtures/tiny/tokenizer.json",
            "epochs": 2,
            "batch_size": 16,
            "weight_decay": 0.0,
            "class_weighting": "none",
        },
        "space": {
            "r": [4],
            "alpha_multiplier": [2],
            "lora_dropout": [0.0],
            "lr": [1e-2, 1e-2],
            "weight_decay": [0.0],
            "epochs": [2],
            "batch_size": [16],
            "warmup_ratio": [0.0],
            "class_weighting": ["none"],
        },
    }


def test_three_trials_write_outputs_and_resume_adds_none(tmp_path: Path) -> None:
    repository = FixtureRepository()
    train = list(repository.load(Split.TRAIN))
    dev = list(repository.load(Split.DEV))
    searcher = OptunaSearcher(
        train,
        dev,
        tmp_path,
        _payload(),
        tracker_factory=lambda run_dir: NullTracker(),
        device=CPU,
    )

    first = searcher.search(_payload(), 3, resume=False)

    assert first.n_trials == 3
    for name in ("search_trials.json", "best_config.toml", "search_summary.md"):
        assert (tmp_path / name).is_file()
    trials = json.loads((tmp_path / "search_trials.json").read_text(encoding="utf-8"))
    assert len(trials["trials"]) == 3
    assert all(trial["state"] == "COMPLETE" for trial in trials["trials"])
    best = tomllib.loads((tmp_path / "best_config.toml").read_text(encoding="utf-8"))
    assert best["base"]["r"] == 4
    assert best["base"]["alpha"] == 8

    second = searcher.search(_payload(), 3, resume=True)

    assert second.n_trials == 0
    after = json.loads((tmp_path / "search_trials.json").read_text(encoding="utf-8"))
    assert len(after["trials"]) == 3


def test_existing_study_without_resume_is_rejected(tmp_path: Path) -> None:
    from stc.domain.errors import InvalidInputError

    repository = FixtureRepository()
    searcher = OptunaSearcher(
        list(repository.load(Split.TRAIN)),
        list(repository.load(Split.DEV)),
        tmp_path,
        _payload(),
        tracker_factory=lambda run_dir: NullTracker(),
        device=CPU,
    )
    searcher.search(_payload(), 1, resume=False)

    with pytest.raises(InvalidInputError, match="--resume"):
        searcher.search(_payload(), 1, resume=False)
