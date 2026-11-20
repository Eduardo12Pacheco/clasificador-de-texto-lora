"""Classifier provider used by the FastAPI adapter (wired by the composition root)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from stc import bootstrap
from stc.application.services.calibrated_classifier import CalibratedClassifier
from stc.config import Settings

__all__ = [
    "BootstrapClassifierProvider",
    "ClassifierProvider",
    "LoadedModel",
]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A loaded calibrated classifier plus the calibration metadata beside it."""

    classifier: CalibratedClassifier
    calibration: Mapping[str, object]


class ClassifierProvider(Protocol):
    """Loads the served model and the optional baseline on demand."""

    def load_default(self) -> LoadedModel:
        """Return the default served model."""
        ...

    def load_baseline(self) -> LoadedModel:
        """Return the baseline model."""
        ...

    def baseline_available(self) -> bool:
        """Return whether a baseline artifact is installed."""
        ...


class BootstrapClassifierProvider:
    """Concrete provider backed by the project's model resolver."""

    def __init__(
        self,
        settings: Settings,
        *,
        default_selector: str | None = None,
    ) -> None:
        self._settings = settings
        self._default_selector = default_selector or settings.model

    def _load(self, selector: str) -> LoadedModel:
        artifact = bootstrap.load_calibrated_classifier(selector, self._settings)
        return LoadedModel(
            classifier=artifact.classifier,
            calibration=artifact.calibration_payload,
        )

    def load_default(self) -> LoadedModel:
        """Load ``STC_MODEL`` or the ``--model`` override."""
        return self._load(self._default_selector)

    def load_baseline(self) -> LoadedModel:
        """Load the baseline artifact."""
        return self._load("baseline")

    def baseline_available(self) -> bool:
        """Return whether the baseline artifact exists."""
        return bootstrap.baseline_available(self._settings)
