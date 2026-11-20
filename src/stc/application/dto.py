"""Typed data-transfer objects that cross application boundaries.

The objects are deliberately framework-free: adapters serialise them and the
domain never sees them, so they can evolve without pulling I/O into the core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import IntegrityError


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
