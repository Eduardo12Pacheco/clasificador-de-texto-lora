"""Typed training configuration shared by the CLI, trainer and search (D7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from stc.domain.errors import InvalidInputError

__all__ = ["TrainingConfig"]

_ALLOWED_METHODS = {"lora", "full"}
_ALLOWED_CLASS_WEIGHTING = {"none", "sqrt_inv_freq"}


def _section(payload: Mapping[str, object]) -> Mapping[str, object]:
    section = payload.get("base", payload)
    if not isinstance(section, Mapping):
        raise InvalidInputError("the [base] section must be a table")
    return section


def _string(section: Mapping[str, object], key: str, default: str | None = None) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value:
        raise InvalidInputError(f"{key} must be a non-empty string")
    return value


def _integer(section: Mapping[str, object], key: str, default: int, *, minimum: int = 0) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidInputError(f"{key} must be an integer >= {minimum}")
    return value


def _number(section: Mapping[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise InvalidInputError(f"{key} must be a non-negative number")
    return float(value)


def _boolean(section: Mapping[str, object], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise InvalidInputError(f"{key} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One fully resolved training run (D7 options included)."""

    backbone: str
    run_dir: Path
    seed: int = 13
    method: str = "lora"
    epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    class_weighting: str = "none"
    grad_accum: int = 1
    max_length: int = 64
    r: int = 16
    alpha: int = 32
    lora_dropout: float = 0.05
    tokenizer_path: Path | None = None
    dataset: str = "unknown"
    git_sha: str = "unknown"
    uv_lock_sha256: str = "unknown"
    save_artifacts: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> TrainingConfig:
        """Validate a TOML-style mapping, accepting a ``[base]`` section."""
        section = _section(payload)
        method = _string(section, "method", "lora")
        if method not in _ALLOWED_METHODS:
            raise InvalidInputError(f"method must be one of {sorted(_ALLOWED_METHODS)}")
        class_weighting = _string(section, "class_weighting", "none")
        if class_weighting not in _ALLOWED_CLASS_WEIGHTING:
            raise InvalidInputError(
                f"class_weighting must be one of {sorted(_ALLOWED_CLASS_WEIGHTING)}"
            )
        run_dir = Path(_string(section, "run_dir"))
        r = _integer(section, "r", 16, minimum=1)
        alpha = _integer(section, "alpha", r * 2, minimum=1)
        tokenizer_path = section.get("tokenizer_path")
        if tokenizer_path is not None and not isinstance(tokenizer_path, str):
            raise InvalidInputError("tokenizer_path must be a string when set")
        return cls(
            backbone=_string(section, "backbone"),
            run_dir=run_dir,
            seed=_integer(section, "seed", 13, minimum=0),
            method=method,
            epochs=_integer(section, "epochs", 4, minimum=1),
            batch_size=_integer(section, "batch_size", 32, minimum=1),
            lr=_number(section, "lr", 3e-4),
            weight_decay=_number(section, "weight_decay", 0.01),
            warmup_ratio=_number(section, "warmup_ratio", 0.06),
            class_weighting=class_weighting,
            grad_accum=_integer(section, "grad_accum", 1, minimum=1),
            max_length=_integer(section, "max_length", 64, minimum=1),
            r=r,
            alpha=alpha,
            lora_dropout=_number(section, "lora_dropout", 0.05),
            tokenizer_path=Path(tokenizer_path) if tokenizer_path is not None else None,
            dataset=_string(section, "dataset", "unknown"),
            git_sha=_string(section, "git_sha", "unknown"),
            uv_lock_sha256=_string(section, "uv_lock_sha256", "unknown"),
            save_artifacts=_boolean(section, "save_artifacts", True),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON/TOML-friendly run configuration."""
        return {
            "backbone": self.backbone,
            "seed": self.seed,
            "method": self.method,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "class_weighting": self.class_weighting,
            "grad_accum": self.grad_accum,
            "max_length": self.max_length,
            "r": self.r,
            "alpha": self.alpha,
            "lora_dropout": self.lora_dropout,
            "tokenizer_path": str(self.tokenizer_path) if self.tokenizer_path else None,
            "dataset": self.dataset,
        }
