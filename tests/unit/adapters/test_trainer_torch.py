"""CPU training loop: artifacts, loss descent, determinism and LoRA coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stc.adapters.data.fixture_source import FixtureRepository
from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.adapters.tracking.null_tracker import NullTracker
from stc.adapters.training.config import TrainingConfig
from stc.adapters.training.model_factory import build_model
from stc.adapters.training.trainer import FullFtTrainer, LoraTrainer
from stc.domain.splits import Split

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


def _splits():
    repository = FixtureRepository()
    return list(repository.load(Split.TRAIN)), list(repository.load(Split.DEV))


def _config(run_dir: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "backbone": "stc-tiny-test",
        "method": "lora",
        "epochs": 2,
        "batch_size": 16,
        "lr": 1e-2,
        "r": 8,
        "alpha": 16,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "class_weighting": "none",
        "grad_accum": 1,
        "max_length": 64,
        "seed": 13,
        "tokenizer_path": "tests/fixtures/tiny/tokenizer.json",
        "run_dir": str(run_dir),
    }
    config.update(overrides)
    return config


def test_lora_run_writes_artifacts_and_reduces_loss(tmp_path: Path) -> None:
    train, dev = _splits()
    run_dir = tmp_path / "seed-13"

    result = LoraTrainer(device=CPU).fit(train, dev, _config(run_dir), NullTracker())

    assert result.metrics["dev_macro_f1"] >= 0.0
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "adapter").is_dir()
    assert (run_dir / "tokenizer.json").is_file()
    assert np.load(run_dir / "dev_logits.npy").shape == (len(dev), 60)
    import json

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["history"][1]["train_loss"] < metrics["history"][0]["train_loss"]


def test_lora_trainable_fraction_and_head(tmp_path: Path) -> None:
    train, dev = _splits()
    run_dir = tmp_path / "run"
    LoraTrainer(device=CPU).fit(train, dev, _config(run_dir, epochs=1), NullTracker())
    import json

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert 0.0 < run["trainable_fraction"] < 0.25
    assert run["trainable_params"] > 0

    tokenizer = HfTokenizer.from_file("tests/fixtures/tiny/tokenizer.json")
    config = TrainingConfig.from_mapping(_config(tmp_path / "probe"))
    model = build_model("stc-tiny-test", 60, "lora", config, tokenizer=tokenizer)
    head = [
        name
        for name, parameter in model.named_parameters()
        if "classifier" in name and parameter.requires_grad
    ]
    assert head


def test_same_seed_twice_gives_identical_dev_logits(tmp_path: Path) -> None:
    train, dev = _splits()
    first = tmp_path / "first"
    second = tmp_path / "second"

    LoraTrainer(device=CPU).fit(train, dev, _config(first, epochs=1), NullTracker())
    LoraTrainer(device=CPU).fit(train, dev, _config(second, epochs=1), NullTracker())

    left = np.load(first / "dev_logits.npy")
    right = np.load(second / "dev_logits.npy")
    assert np.max(np.abs(left - right)) <= 1e-6


def test_full_fine_tuning_path_runs(tmp_path: Path) -> None:
    train, dev = _splits()
    run_dir = tmp_path / "full"

    result = FullFtTrainer(device=CPU).fit(
        train,
        dev,
        _config(run_dir, method="full", epochs=1, lr=1e-2),
        NullTracker(),
    )

    assert result.info.kind == "torch-full"
    assert (run_dir / "model").is_dir()
    assert (run_dir / "metrics.json").is_file()


def test_class_weighting_path_runs(tmp_path: Path) -> None:
    train, dev = _splits()

    result = LoraTrainer(device=CPU).fit(
        train,
        dev,
        _config(tmp_path / "weighted", epochs=1, class_weighting="sqrt_inv_freq"),
        NullTracker(),
    )

    assert "dev_macro_f1" in result.metrics
