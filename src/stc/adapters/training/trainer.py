"""Custom, explicit PyTorch training loops for LoRA and full fine-tuning (D7)."""

from __future__ import annotations

import json
import math
import platform
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from stc import __version__
from stc.adapters.text.hf_tokenizer import HfTokenizer, resolve_tokenizer_file
from stc.adapters.training.batching import shuffled_batches
from stc.adapters.training.checkpoints import RunDir
from stc.adapters.training.class_weighting import sqrt_inv_freq_weights
from stc.adapters.training.config import TrainingConfig
from stc.adapters.training.model_factory import (
    build_model,
    device_for_training,
    parameter_counts,
)
from stc.adapters.training.seed import set_all_seeds
from stc.application.dto import ModelInfo, TrainedRun
from stc.application.ports import ExperimentTracker
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.metrics import accuracy, confusion_matrix, macro_f1, present_labels

__all__ = ["EpochCallback", "FullFtTrainer", "LoraTrainer", "evaluate_logits"]

EpochCallback = Callable[[int, float], None]


def _versions() -> dict[str, str]:
    import peft
    import transformers

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "numpy": str(np.__version__),
    }


def _nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(nn.functional.cross_entropy(logits.float(), targets).item())


def evaluate_logits(
    logits: NDArray[np.float32],
    y_true: Sequence[int],
) -> dict[str, float]:
    """Compute the D2 dev metrics from a logits matrix."""
    matrix = np.asarray(logits, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.argmax(matrix, axis=1).astype(np.int64)
    cm = confusion_matrix(true, predicted, matrix.shape[1])
    labels = present_labels(true)
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    nll = float(-np.mean(log_probs[np.arange(true.size), true]))
    return {
        "accuracy": accuracy(true, predicted),
        "macro_f1": macro_f1(cm, labels),
        "nll": nll,
    }


class _BaseTrainer:
    """Shared loop; subclasses only choose the adaptation method."""

    method: str = "lora"

    def __init__(
        self,
        *,
        save_artifacts: bool = True,
        on_epoch_end: EpochCallback | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._save_artifacts = save_artifacts
        self._on_epoch_end = on_epoch_end
        self._device = device_for_training() if device is None else device

    def fit(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        config: Mapping[str, object],
        tracker: ExperimentTracker,
    ) -> TrainedRun:
        """Fit on train, evaluate dev after every epoch and persist the run."""
        if not train:
            raise InvalidInputError("training needs a non-empty train split")
        if not dev:
            raise InvalidInputError("training needs a non-empty dev split")
        resolved = replace(TrainingConfig.from_mapping(config), method=self.method)
        return self._fit_with_oom_retry(train, dev, resolved, tracker)

    def _fit_with_oom_retry(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        config: TrainingConfig,
        tracker: ExperimentTracker,
    ) -> TrainedRun:
        try:
            return self._fit_once(train, dev, config, tracker)
        except torch.cuda.OutOfMemoryError:
            if self._device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            retried = replace(config, batch_size=16, grad_accum=2)
            return self._fit_once(train, dev, retried, tracker, oom_retry=True)

    def _fit_once(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        config: TrainingConfig,
        tracker: ExperimentTracker,
        *,
        oom_retry: bool = False,
    ) -> TrainedRun:
        set_all_seeds(config.seed)
        run_dir = RunDir(config.run_dir)
        run_dir.create()
        tokenizer_file = resolve_tokenizer_file(config.backbone, config.tokenizer_path)
        tokenizer = HfTokenizer.from_file(tokenizer_file, max_length=config.max_length)
        labels = LabelSpace.from_intents(sorted({row.intent for row in (*train, *dev)}))
        model = build_model(
            config.backbone,
            labels.n_intents,
            self.method,
            config,
            tokenizer=tokenizer,
        )
        model.to(self._device)
        trainable, total = parameter_counts(model)
        dev_examples = list(dev)
        dev_ids = [labels.intent_index(row.intent) for row in dev_examples]
        criterion = nn.CrossEntropyLoss(weight=self._class_weights(train, labels, config))
        if criterion.weight is not None:
            criterion.weight = criterion.weight.to(self._device)
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )
        steps_per_epoch = max(
            1,
            math.ceil(len(train) / (config.batch_size * config.grad_accum)),
        )
        total_steps = max(1, steps_per_epoch * config.epochs)
        scheduler = LambdaLR(optimizer, lr_lambda=_linear_warmup_decay(total_steps, config))

        tracker.start_run(
            run_dir.path.name,
            tags={
                "backbone": config.backbone,
                "method": self.method,
                "dataset": config.dataset,
                "git_sha": config.git_sha,
                "seed": str(config.seed),
            },
        )
        tracker.log_params({**config.to_dict(), "device": str(self._device)})
        history: list[dict[str, float]] = []
        epoch_seconds: list[float] = []
        bf16 = self._device.type == "cuda" and torch.cuda.is_bf16_supported()
        started = time.monotonic()
        try:
            for epoch in range(1, config.epochs + 1):
                epoch_start = time.monotonic()
                model.train()
                batches = shuffled_batches(
                    train,
                    config.batch_size,
                    seed=config.seed,
                    epoch=epoch,
                )
                running_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for step, batch in enumerate(batches, start=1):
                    input_ids, attention_mask, targets = self._encode_batch(
                        tokenizer, batch, labels
                    )
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if bf16
                        else nullcontext()
                    )
                    with autocast:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = criterion(logits.float(), targets)
                    running_loss += float(loss.item())
                    (loss / config.grad_accum).backward()
                    if step % config.grad_accum == 0 or step == len(batches):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                dev_logits, dev_metrics = self._evaluate(model, tokenizer, dev_examples, dev_ids)
                epoch_metrics = {
                    "train_loss": running_loss / max(1, len(batches)),
                    **dev_metrics,
                }
                history.append({"epoch": float(epoch), **epoch_metrics})
                tracker.log_metrics(epoch_metrics, step=epoch)
                epoch_seconds.append(time.monotonic() - epoch_start)
                if self._on_epoch_end is not None:
                    self._on_epoch_end(epoch, dev_metrics["macro_f1"])
        finally:
            tracker.end_run()
        wall_seconds = time.monotonic() - started
        peak_vram = (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if self._device.type == "cuda"
            else 0.0
        )
        self._write_artifacts(
            run_dir,
            model,
            tokenizer_file,
            config,
            dev_logits,
            history,
            epoch_seconds,
            wall_seconds,
            peak_vram,
            trainable,
            total,
            len(train),
            len(dev),
            labels,
            oom_retry,
        )
        final = history[-1]
        metrics = {
            "dev_accuracy": final["accuracy"],
            "dev_macro_f1": final["macro_f1"],
            "dev_nll": final["nll"],
            "n_train": float(len(train)),
            "n_dev": float(len(dev)),
            "wall_seconds": wall_seconds,
            "peak_vram_mib": peak_vram,
            "trainable_params": float(trainable),
            "total_params": float(total),
        }
        info = ModelInfo(
            model_id=config.backbone,
            kind=f"torch-{self.method}",
            version=__version__,
            quantization=None,
            parameters=total,
            license="Apache-2.0" if "MrBERT" in config.backbone else "unknown",
            n_intents=labels.n_intents,
            n_scenarios=labels.n_scenarios,
        )
        return TrainedRun(run_dir=run_dir.path, info=info, metrics=metrics)

    def _class_weights(
        self,
        train: Sequence[Example],
        labels: LabelSpace,
        config: TrainingConfig,
    ) -> torch.Tensor | None:
        if config.class_weighting == "none":
            return None
        indices = np.asarray(
            [labels.intent_index(row.intent) for row in train],
            dtype=np.int64,
        )
        counts = np.bincount(indices, minlength=labels.n_intents)
        weights = sqrt_inv_freq_weights(counts.tolist())
        return torch.as_tensor(weights, dtype=torch.float32)

    def _encode_batch(
        self,
        tokenizer: HfTokenizer,
        batch: Sequence[Example],
        labels: LabelSpace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = tokenizer.encode_batch([row.text for row in batch])
        input_ids = torch.as_tensor(encoded.input_ids, dtype=torch.long, device=self._device)
        attention_mask = torch.as_tensor(
            encoded.attention_mask, dtype=torch.long, device=self._device
        )
        targets = torch.as_tensor(
            [labels.intent_index(row.intent) for row in batch],
            dtype=torch.long,
            device=self._device,
        )
        return input_ids, attention_mask, targets

    def _evaluate(
        self,
        model: nn.Module,
        tokenizer: HfTokenizer,
        dev: Sequence[Example],
        dev_ids: Sequence[int],
    ) -> tuple[NDArray[np.float32], dict[str, float]]:
        model.eval()
        chunks: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(dev), 64):
                batch = dev[start : start + 64]
                encoded = tokenizer.encode_batch([row.text for row in batch])
                input_ids = torch.as_tensor(
                    encoded.input_ids, dtype=torch.long, device=self._device
                )
                attention_mask = torch.as_tensor(
                    encoded.attention_mask, dtype=torch.long, device=self._device
                )
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                chunks.append(logits.float().cpu().numpy())
        dev_logits = np.concatenate(chunks, axis=0).astype(np.float32)
        metrics = evaluate_logits(dev_logits, dev_ids)
        return dev_logits, metrics

    def _write_artifacts(
        self,
        run_dir: RunDir,
        model: nn.Module,
        tokenizer_file: Path,
        config: TrainingConfig,
        dev_logits: NDArray[np.float32],
        history: Sequence[Mapping[str, float]],
        epoch_seconds: Sequence[float],
        wall_seconds: float,
        peak_vram: float,
        trainable: int,
        total: int,
        n_train: int,
        n_dev: int,
        labels: LabelSpace,
        oom_retry: bool,
    ) -> None:
        np.save(run_dir.dev_logits, dev_logits)
        shutil.copyfile(tokenizer_file, run_dir.tokenizer_json)
        if self._save_artifacts:
            if config.method == "lora":
                self._save_peft(model, run_dir.adapter_dir)
            else:
                model.save_pretrained(run_dir.model_dir)
        run_payload: dict[str, object] = {
            "schema_version": 1,
            "stc_version": __version__,
            "backbone": config.backbone,
            "method": config.method,
            "seed": config.seed,
            "dataset": config.dataset,
            "config": config.to_dict(),
            "library_versions": _versions(),
            "device": str(self._device),
            "gpu_name": (torch.cuda.get_device_name(0) if self._device.type == "cuda" else "cpu"),
            "git_sha": config.git_sha,
            "uv_lock_sha256": config.uv_lock_sha256,
            "n_train": n_train,
            "n_dev": n_dev,
            "n_intents": labels.n_intents,
            "n_scenarios": labels.n_scenarios,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_fraction": trainable / total if total else 0.0,
            "epoch_seconds": list(epoch_seconds),
            "wall_seconds": wall_seconds,
            "peak_vram_mib": peak_vram,
            "oom_retry": oom_retry,
        }
        metrics_payload: dict[str, object] = {
            "schema_version": 1,
            "split": "dev",
            "history": list(history),
            "dev_accuracy": history[-1]["accuracy"],
            "dev_macro_f1": history[-1]["macro_f1"],
            "dev_nll": history[-1]["nll"],
            "n_dev": n_dev,
        }
        run_dir.run_json.write_text(
            json.dumps(run_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        run_dir.metrics_json.write_text(
            json.dumps(metrics_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _save_peft(model: nn.Module, destination: Path) -> None:
        save = getattr(model, "save_pretrained", None)
        if save is None:
            raise InvalidInputError("LoRA model cannot be saved (missing save_pretrained)")
        destination.mkdir(parents=True, exist_ok=True)
        save(destination)


def _linear_warmup_decay(total_steps: int, config: TrainingConfig) -> Callable[[int], float]:
    warmup_steps = round(config.warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0
        return max(0.0, (total_steps - step) / remaining)

    return factor


class LoraTrainer(_BaseTrainer):
    """LoRA fine-tuning with PEFT (D7)."""

    def __init__(
        self,
        *,
        save_artifacts: bool = True,
        on_epoch_end: EpochCallback | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            save_artifacts=save_artifacts,
            on_epoch_end=on_epoch_end,
            device=device,
        )


class FullFtTrainer(_BaseTrainer):
    """Full fine-tuning reference trainer (D7)."""

    method = "full"

    def __init__(
        self,
        *,
        save_artifacts: bool = True,
        on_epoch_end: EpochCallback | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            save_artifacts=save_artifacts,
            on_epoch_end=on_epoch_end,
            device=device,
        )
