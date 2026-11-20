"""Typed data-transfer objects that cross application boundaries.

The objects are deliberately framework-free: adapters serialise them and the
domain never sees them, so they can evolve without pulling I/O into the core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import IntegrityError, InvalidInputError


def _int_mapping(payload: object, field_name: str) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise IntegrityError(f"{field_name} must be an object")
    result: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool):
            raise IntegrityError(f"{field_name} must map strings to integers")
        result[key] = value
    return result


def _string(payload: object, field_name: str) -> str:
    if not isinstance(payload, str):
        raise IntegrityError(f"{field_name} must be a string")
    return payload


@dataclass(frozen=True, slots=True)
class RawDatasetRef:
    """A verified, extracted raw dataset on disk."""

    root: Path
    sha256: str
    size_bytes: int
    source_url: str
    cached: bool = False


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    """Provenance and row counts persisted next to the processed splits."""

    dataset_id: str
    version: str
    tarball_sha256: str
    stc_version: str
    raw_counts: Mapping[str, int]
    processed_counts: Mapping[str, int]
    dropped_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation with stable keys."""
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "tarball_sha256": self.tarball_sha256,
            "stc_version": self.stc_version,
            "raw_counts": dict(self.raw_counts),
            "processed_counts": dict(self.processed_counts),
            "dropped_counts": dict(self.dropped_counts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetMeta:
        """Rebuild metadata, rejecting malformed persisted values."""
        return cls(
            dataset_id=_string(payload.get("dataset_id"), "dataset_id"),
            version=_string(payload.get("version"), "version"),
            tarball_sha256=_string(payload.get("tarball_sha256"), "tarball_sha256"),
            stc_version=_string(payload.get("stc_version"), "stc_version"),
            raw_counts=_int_mapping(payload.get("raw_counts"), "raw_counts"),
            processed_counts=_int_mapping(payload.get("processed_counts"), "processed_counts"),
            dropped_counts=_int_mapping(payload.get("dropped_counts"), "dropped_counts"),
        )


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    """Right-truncated token ids and their attention mask."""

    input_ids: NDArray[np.int64]
    attention_mask: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Cheap, immutable description of a classifier artifact."""

    model_id: str
    kind: str
    version: str
    quantization: str | None
    parameters: int | None
    license: str
    n_intents: int
    n_scenarios: int


@dataclass(frozen=True, slots=True)
class TrainedRun:
    """A finished training run and the metrics measured on dev."""

    run_dir: Path
    info: ModelInfo
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The outcome of a hyper-parameter search."""

    search_dir: Path
    best_params: Mapping[str, object]
    best_value: float
    n_trials: int


@dataclass(frozen=True, slots=True)
class ExportReport:
    """Export parity and quantisation evidence for one shipped artifact."""

    out_dir: Path
    parity: Mapping[str, object]
    quantization: Mapping[str, object]


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise InvalidInputError(f"{field_name} must be a list")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise InvalidInputError(f"{field_name} must be a positive number")
    return float(value)


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInputError(f"{field_name} must be a positive integer")
    return value


def _ngram_pairs(value: object, field_name: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in _sequence(value, field_name):
        values = _sequence(item, field_name)
        if len(values) != 2:
            raise InvalidInputError(f"{field_name} entries must be [min, max] pairs")
        lower = _positive_int(values[0], field_name)
        upper = _positive_int(values[1], field_name)
        if lower > upper:
            raise InvalidInputError(f"{field_name} lower bound must not exceed the upper bound")
        pairs.append((lower, upper))
    if not pairs:
        raise InvalidInputError(f"{field_name} must contain at least one pair")
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class BaselineGrid:
    """The D5 grid, kept as data so experiments stay reproducible.

    ``class_weights`` uses ``"none"`` for the unweighted variant because TOML
    has no null literal.
    """

    word_ngrams: tuple[tuple[int, int], ...] = ((1, 2), (1, 3))
    char_ngrams: tuple[tuple[int, int], ...] = ((2, 5), (3, 6))
    svc_c: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
    class_weights: tuple[str, ...] = ("none", "balanced")
    logreg_c: tuple[float, ...] = (10.0, 30.0)
    logreg_max_iter: int = 2000
    random_state: int = 13
    quick_configs: int = 2

    @classmethod
    def default(cls) -> BaselineGrid:
        """Return the grid fixed by decision D5."""
        return cls()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> BaselineGrid:
        """Validate a TOML ``[grid]`` table, falling back to D5 defaults."""
        base = cls()
        word_ngrams = (
            _ngram_pairs(payload["word_ngrams"], "word_ngrams")
            if "word_ngrams" in payload
            else base.word_ngrams
        )
        char_ngrams = (
            _ngram_pairs(payload["char_ngrams"], "char_ngrams")
            if "char_ngrams" in payload
            else base.char_ngrams
        )
        svc_c = (
            tuple(_positive_float(value, "svc_c") for value in _sequence(payload["svc_c"], "svc_c"))
            if "svc_c" in payload
            else base.svc_c
        )
        logreg_c = (
            tuple(
                _positive_float(value, "logreg_c")
                for value in _sequence(payload["logreg_c"], "logreg_c")
            )
            if "logreg_c" in payload
            else base.logreg_c
        )
        if not svc_c:
            raise InvalidInputError("svc_c must contain at least one value")
        if not logreg_c:
            raise InvalidInputError("logreg_c must contain at least one value")
        class_weights = (
            tuple(str(value) for value in _sequence(payload["class_weights"], "class_weights"))
            if "class_weights" in payload
            else base.class_weights
        )
        if not class_weights or any(weight not in {"none", "balanced"} for weight in class_weights):
            raise InvalidInputError("class_weights values must be 'none' or 'balanced'")
        random_state = (
            _positive_int(payload["random_state"], "random_state")
            if "random_state" in payload
            else base.random_state
        )
        logreg_max_iter = (
            _positive_int(payload["logreg_max_iter"], "logreg_max_iter")
            if "logreg_max_iter" in payload
            else base.logreg_max_iter
        )
        quick_configs = (
            _positive_int(payload["quick_configs"], "quick_configs")
            if "quick_configs" in payload
            else base.quick_configs
        )
        return cls(
            word_ngrams=word_ngrams,
            char_ngrams=char_ngrams,
            svc_c=svc_c,
            class_weights=class_weights,
            logreg_c=logreg_c,
            logreg_max_iter=logreg_max_iter,
            random_state=random_state,
            quick_configs=quick_configs,
        )

    @classmethod
    def from_toml_table(cls, payload: Mapping[str, object]) -> BaselineGrid:
        """Accept either a flat table or a document with a ``[grid]`` section."""
        section = payload.get("grid", payload)
        if not isinstance(section, Mapping):
            raise InvalidInputError("the [grid] section must be a table")
        return cls.from_mapping(section)

    @property
    def n_svc_configs(self) -> int:
        """Number of LinearSVC configurations in the full grid."""
        return (
            len(self.word_ngrams)
            * len(self.char_ngrams)
            * len(self.svc_c)
            * len(self.class_weights)
        )

    @property
    def n_configs(self) -> int:
        """Number of configurations in the full grid, logistic ones included."""
        return self.n_svc_configs + len(self.logreg_c)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON/TOML-friendly form of the grid."""
        return {
            "word_ngrams": [list(pair) for pair in self.word_ngrams],
            "char_ngrams": [list(pair) for pair in self.char_ngrams],
            "svc_c": list(self.svc_c),
            "class_weights": list(self.class_weights),
            "logreg_c": list(self.logreg_c),
            "logreg_max_iter": self.logreg_max_iter,
            "random_state": self.random_state,
            "quick_configs": self.quick_configs,
        }


@dataclass(frozen=True, slots=True)
class BaselineCandidate:
    """Dev metrics for one fitted baseline configuration."""

    params: Mapping[str, object]
    dev_accuracy: float
    dev_macro_f1: float
    n_features: int
    fit_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable candidate row."""
        return {
            "params": dict(self.params),
            "dev_accuracy": self.dev_accuracy,
            "dev_macro_f1": self.dev_macro_f1,
            "n_features": self.n_features,
            "fit_seconds": self.fit_seconds,
        }
