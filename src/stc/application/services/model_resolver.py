"""Resolve ``--model`` values to an artifact directory (D13/D19).

Accepted forms are an existing path, the reserved name ``baseline`` and an
installed model name (optionally versioned). Registry downloads happen before
this resolver runs; a miss always names the command that fixes it.
"""

from __future__ import annotations

from pathlib import Path

from stc.application.ports import ModelStore
from stc.domain.errors import ArtifactNotFoundError

__all__ = ["BASELINE_NAME", "ModelResolver"]

BASELINE_NAME = "baseline"


class ModelResolver:
    """Map a CLI model selector to an on-disk artifact."""

    def __init__(
        self,
        store: ModelStore,
        *,
        default_name: str,
        baseline_dir: Path | None = None,
        default_version: str | None = None,
    ) -> None:
        self._store = store
        self._default_name = default_name
        self._baseline_dir = Path(baseline_dir) if baseline_dir is not None else None
        self._default_version = default_version

    def resolve(self, model: str, version: str | None = None) -> Path:
        """Return the artifact directory for ``model``."""
        candidate = Path(model)
        if candidate.exists():
            return candidate
        if model == BASELINE_NAME:
            if self._baseline_dir is not None and self._baseline_dir.is_dir():
                return self._baseline_dir
            installed = self._installed_version(self._default_name, self._default_version)
            if installed is not None:
                baseline = self._store.load(self._default_name, installed) / BASELINE_NAME
                if baseline.is_dir():
                    return baseline
            raise ArtifactNotFoundError(
                "baseline model not found; run: stc train baseline (or install a bundle "
                "with a baseline/ directory via stc model pull)"
            )
        resolved_version = version or self._installed_version(model, self._default_version)
        if resolved_version is None:
            raise ArtifactNotFoundError(
                f"model not found: {model}; run: stc model pull {model} "
                "(or stc train baseline for a local baseline)"
            )
        return self._store.load(model, resolved_version)

    def _installed_version(self, name: str, explicit: str | None) -> str | None:
        if explicit is not None:
            return explicit
        versions = sorted(
            info.version for info in self._store.list_installed() if info.model_id == name
        )
        return versions[-1] if versions else None
