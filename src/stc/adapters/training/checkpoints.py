"""The on-disk layout of one training run (T7/T8 contract)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["RunDir"]


@dataclass(frozen=True, slots=True)
class RunDir:
    """Paths of every artifact a finished run writes."""

    path: Path

    @property
    def run_json(self) -> Path:
        """Run metadata, config, versions and timings."""
        return self.path / "run.json"

    @property
    def metrics_json(self) -> Path:
        """Final dev metrics and per-epoch history."""
        return self.path / "metrics.json"

    @property
    def dev_logits(self) -> Path:
        """Float32 dev logits with shape ``(n_dev, n_labels)``."""
        return self.path / "dev_logits.npy"

    @property
    def tokenizer_json(self) -> Path:
        """A copy of the backbone tokenizer used by the run."""
        return self.path / "tokenizer.json"

    @property
    def adapter_dir(self) -> Path:
        """PEFT adapter directory for LoRA runs."""
        return self.path / "adapter"

    @property
    def model_dir(self) -> Path:
        """Full fine-tuned model directory."""
        return self.path / "model"

    def exists(self) -> bool:
        """Return true when a finished run already occupies this directory."""
        return self.run_json.is_file()

    def create(self) -> None:
        """Create the directory tree, tolerating an existing run."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.adapter_dir.parent.mkdir(parents=True, exist_ok=True)
