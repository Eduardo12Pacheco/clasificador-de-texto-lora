"""Backbone x seed screening with the pre-registered D6 recipe."""

from __future__ import annotations

import gc
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.adapters.training.checkpoints import RunDir
from stc.adapters.training.config import TrainingConfig
from stc.adapters.training.model_factory import TINY_BACKBONE, build_base_model
from stc.adapters.training.trainer import LoraTrainer
from stc.application.dto import ScreeningRun
from stc.application.ports import ExperimentTracker
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example

__all__ = ["ScreeningConfig", "ScreeningRunner", "measure_cpu_latency"]

TrackerFactory = Callable[[Path], ExperimentTracker]

_NEVER_EXPORTABLE = "Qwen/Qwen3-0.6B"
_DEFAULT_BACKBONES = (
    "BSC-LT/MrBERT-es",
    "bertin-project/bertin-roberta-base-spanish",
    "Qwen/Qwen3-0.6B",
)


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    """The screening recipe from D7 plus its candidate backbones."""

    base: Mapping[str, object]
    backbones: tuple[str, ...]
    seeds: tuple[int, ...]
    learning_rates: Mapping[str, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ScreeningConfig:
        """Validate ``configs/screening.toml``."""
        base = payload.get("base", {})
        screening = payload.get("screening", payload)
        if not isinstance(base, Mapping) or not isinstance(screening, Mapping):
            raise InvalidInputError("[base] and [screening] must be tables")
        backbones = screening.get("backbones", list(_DEFAULT_BACKBONES))
        if not isinstance(backbones, list) or not all(isinstance(item, str) for item in backbones):
            raise InvalidInputError("screening.backbones must be a list of strings")
        seeds = screening.get("seeds", [13, 42])
        if not isinstance(seeds, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in seeds
        ):
            raise InvalidInputError("screening.seeds must be a list of integers")
        rates = screening.get("learning_rates", {})
        if not isinstance(rates, Mapping) or not all(
            isinstance(value, int | float) and not isinstance(value, bool)
            for value in rates.values()
        ):
            raise InvalidInputError("screening.learning_rates must map backbones to numbers")
        return cls(
            base=dict(base),
            backbones=tuple(str(item) for item in backbones),
            seeds=tuple(int(item) for item in seeds),
            learning_rates={str(key): float(value) for key, value in rates.items()},
        )

    def learning_rate_for(self, backbone: str) -> float:
        """Return the backbone-specific learning rate (3e-4 default)."""
        return self.learning_rates.get(backbone, 3e-4)


def _slug(backbone: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in backbone).strip("-").lower()


def _latency_texts(dev: Sequence[Example], limit: int = 50) -> list[str]:
    ordered = sorted(dev, key=lambda row: row.id)[:limit]
    return [row.text for row in ordered]


def measure_cpu_latency(
    run_dir: RunDir,
    config: TrainingConfig,
    dev: Sequence[Example],
    n_labels: int,
    *,
    threads: int = 4,
    limit: int = 50,
) -> float:
    """Median single-utterance CPU latency in milliseconds (fp32)."""
    torch.set_num_threads(max(1, threads))
    tokenizer = HfTokenizer.from_file(config.run_dir / "tokenizer.json")
    model = build_base_model(config.backbone, n_labels, tokenizer=tokenizer)
    if config.method == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, run_dir.adapter_dir)
    else:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(run_dir.model_dir)
    model.eval()
    model.to(torch.device("cpu"))
    texts = _latency_texts(dev, limit)
    durations: list[float] = []
    with torch.inference_mode():
        for text in texts:
            encoded = tokenizer.encode_batch([text])
            input_ids = torch.as_tensor(encoded.input_ids, dtype=torch.long)
            attention_mask = torch.as_tensor(encoded.attention_mask, dtype=torch.long)
            started = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            durations.append((time.perf_counter() - started) * 1000.0)
    return float(statistics.median(durations)) if durations else 0.0


class ScreeningRunner:
    """Train every backbone x seed pair, skipping finished runs (resumable)."""

    def __init__(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        out_dir: Path,
        config: ScreeningConfig,
        *,
        tracker_factory: TrackerFactory,
        device: torch.device | None = None,
    ) -> None:
        self._train = list(train)
        self._dev = list(dev)
        self._out_dir = Path(out_dir)
        self._config = config
        self._tracker_factory = tracker_factory
        self._device = device

    def run(self) -> list[ScreeningRun]:
        """Run screening in backbone-major, seed-minor order."""
        results: list[ScreeningRun] = []
        for backbone in self._config.backbones:
            for seed in self._config.seeds:
                try:
                    results.append(self._run_one(backbone, seed))
                finally:
                    # Candidate backbones are large (Qwen3-0.6B is a decoder);
                    # release the previous model before loading the next one.
                    gc.collect()
                    torch.cuda.empty_cache()
        return results

    def _run_one(self, backbone: str, seed: int) -> ScreeningRun:
        run_dir = RunDir(self._out_dir / f"{_slug(backbone)}-seed-{seed}")
        config_payload: dict[str, object] = {
            **self._config.base,
            "backbone": backbone,
            "seed": seed,
            "lr": self._config.learning_rate_for(backbone),
            "method": "lora",
            "run_dir": str(run_dir.path),
            "dataset": self._config.base.get("dataset", "unknown"),
        }
        if not run_dir.exists():
            trainer = LoraTrainer(device=self._device)
            trainer.fit(
                self._train,
                self._dev,
                config_payload,
                self._tracker_factory(run_dir.path),
            )
        run_json = _read_json(run_dir.run_json)
        metrics = _read_json(run_dir.metrics_json)
        n_labels = _integer_field(run_json, "n_intents")
        resolved = TrainingConfig.from_mapping(config_payload)
        latency = measure_cpu_latency(run_dir, resolved, self._dev, n_labels)
        epoch_seconds = run_json.get("epoch_seconds", [])
        mean_epoch_seconds = (
            float(sum(epoch_seconds) / len(epoch_seconds))
            if isinstance(epoch_seconds, list) and epoch_seconds
            else 0.0
        )
        return ScreeningRun(
            backbone=backbone,
            seed=seed,
            dev_macro_f1=_number_field(metrics, "dev_macro_f1"),
            dev_accuracy=_number_field(metrics, "dev_accuracy"),
            trainable_params=_integer_field(run_json, "trainable_params"),
            epoch_seconds=mean_epoch_seconds,
            peak_vram_mib=_number_field(run_json, "peak_vram_mib"),
            cpu_latency_ms_p50=latency,
            exportable=backbone != _NEVER_EXPORTABLE and backbone != TINY_BACKBONE,
            run_dir=run_dir.path,
            oom_retry=bool(run_json.get("oom_retry", False)),
        )


def _number_field(payload: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _integer_field(payload: Mapping[str, object], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidInputError(f"cannot read screening artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise InvalidInputError(f"screening artifact {path} must be a JSON object")
    return payload
