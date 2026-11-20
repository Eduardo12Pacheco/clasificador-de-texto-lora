"""Ports (typing protocols) that the application layer depends on.

Adapters implement these protocols; the application layer only imports this
module and the pure domain, which keeps the dependency arrow pointing inwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from stc.application.dto import (
    DatasetMeta,
    EncodedBatch,
    ExportReport,
    ModelInfo,
    RawDatasetRef,
    SearchResult,
    TrainedRun,
)
from stc.domain.access import Clock, TestAccessEntry
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.splits import Split

__all__ = [
    "AccessLog",
    "Classifier",
    "ClassifierTrainer",
    "Clock",
    "DatasetParser",
    "DatasetRepository",
    "ExperimentTracker",
    "HyperparameterSearcher",
    "ModelExporter",
    "ModelHub",
    "ModelStore",
    "RawDatasetSource",
    "ReportWriter",
    "Tokenizer",
]


class RawDatasetSource(Protocol):
    """Downloads, verifies and extracts the pinned raw dataset."""

    def fetch(self, dest: Path, *, force: bool = False) -> RawDatasetRef:
        """Return a verified raw dataset rooted at ``dest``."""
        ...

    def verify(self, dest: Path) -> str:
        """Re-verify a cached dataset and return its sha256."""
        ...


class DatasetParser(Protocol):
    """Parses a raw dataset file into the three official splits."""

    def parse(self, path: Path) -> Mapping[Split, tuple[Example, ...]]:
        """Read ``path`` and validate every row it contains."""
        ...


class DatasetRepository(Protocol):
    """Persistence of processed splits and their provenance metadata."""

    def load(self, split: Split) -> Sequence[Example]:
        """Read one processed split."""
        ...

    def save(self, split: Split, examples: Sequence[Example]) -> None:
        """Persist one split deterministically."""
        ...

    def meta(self) -> DatasetMeta:
        """Read the provenance metadata written by the prepare pipeline."""
        ...

    def save_meta(self, meta: DatasetMeta) -> None:
        """Persist provenance metadata."""
        ...


class Tokenizer(Protocol):
    """Encodes batches of utterances with the backbone tokenizer."""

    @property
    def max_length(self) -> int:
        """Maximum sequence length in tokens."""
        ...

    def encode_batch(self, texts: Sequence[str]) -> EncodedBatch:
        """Encode texts into right-truncated, dynamically padded id arrays."""
        ...


class Classifier(Protocol):
    """Anything that scores utterances over a fixed label space."""

    @property
    def info(self) -> ModelInfo:
        """Describe the underlying artifact."""
        ...

    @property
    def labels(self) -> LabelSpace:
        """The label space the scores are ordered by."""
        ...

    def predict_scores(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one row of logits per utterance."""
        ...


class ClassifierTrainer(Protocol):
    """Trains a classifier and records the run through a tracker."""

    def fit(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        config: Mapping[str, object],
        tracker: ExperimentTracker,
    ) -> TrainedRun:
        """Fit on train, evaluate on dev and return the finished run."""
        ...


class HyperparameterSearcher(Protocol):
    """Search hyper-parameters, optionally resuming a previous study."""

    def search(
        self,
        space: Mapping[str, object],
        n_trials: int,
        *,
        resume: bool = False,
    ) -> SearchResult:
        """Run or resume a search and return its best trial."""
        ...


class ModelExporter(Protocol):
    """Exports a trained run to a portable artifact."""

    def export(self, run_dir: Path, out_dir: Path, *, quantize: str = "none") -> ExportReport:
        """Export ``run_dir`` into ``out_dir``."""
        ...


class ModelStore(Protocol):
    """Local storage of installed model bundles."""

    def save(self, bundle_dir: Path, name: str, version: str) -> Path:
        """Install a bundle directory into the store."""
        ...

    def load(self, name: str, version: str) -> Path:
        """Return the directory of an installed bundle."""
        ...

    def list_installed(self) -> Sequence[ModelInfo]:
        """List installed bundles."""
        ...

    def verify(self, bundle_dir: Path) -> None:
        """Verify a bundle's manifest and file checksums."""
        ...


class ModelHub(Protocol):
    """Remote distribution of model bundles."""

    def pull(self, entry: Mapping[str, object], dest: Path) -> Path:
        """Download and verify one release entry into ``dest``."""
        ...


class ExperimentTracker(Protocol):
    """Minimal experiment-tracking interface used by trainers."""

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        """Start a run."""
        ...

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record configuration parameters."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record metrics for one step."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Attach a file or directory to the run."""
        ...

    def end_run(self) -> None:
        """Finalise the run."""
        ...


class ReportWriter(Protocol):
    """Writes deterministic reports and figures."""

    def write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Write compact, key-sorted JSON."""
        ...

    def write_csv(
        self,
        path: Path,
        header: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> None:
        """Write a CSV with the given header."""
        ...

    def write_markdown(self, path: Path, text: str) -> None:
        """Write a Markdown document."""
        ...

    def write_figure(self, path: Path, figure: object) -> None:
        """Render a figure object to an image file."""
        ...


class AccessLog(Protocol):
    """Append-only audit trail of authorised test-split reads."""

    def record(self, entry: TestAccessEntry) -> None:
        """Append one JSON line describing an audited read."""
        ...
