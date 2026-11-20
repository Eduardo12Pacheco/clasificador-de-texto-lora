"""Skops persistence with a strict type allow-list and a hashed manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import skops.io as skops_io
from numpy.typing import NDArray

from stc import __version__
from stc.application.dto import ModelInfo
from stc.application.ports import Classifier
from stc.domain.errors import (
    ArtifactNotFoundError,
    IntegrityError,
    InvalidInputError,
    UntrustedArtifactError,
)
from stc.domain.labels import LabelSpace

__all__ = [
    "ALLOWED_MODULE_PREFIXES",
    "MAX_BASELINE_BYTES",
    "SkopsBaselineStore",
    "cast_linear_coefficients",
    "is_allowed_type",
    "load",
    "save",
    "sha256_file",
]

ALLOWED_MODULE_PREFIXES = ("sklearn.", "numpy.", "scipy.sparse.", "builtins.")
MAX_BASELINE_BYTES = 40 * 1024 * 1024
MANIFEST_KIND = "tfidf-linear"


def is_allowed_type(qualified_name: str) -> bool:
    """Accept only sklearn, numpy, scipy.sparse and builtins types."""
    if not qualified_name or "." not in qualified_name:
        return False
    return qualified_name.startswith(ALLOWED_MODULE_PREFIXES)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(pipeline: object, path: Path) -> None:
    """Persist one fitted sklearn pipeline with skops."""
    path.parent.mkdir(parents=True, exist_ok=True)
    skops_io.dump(pipeline, path)


def load(path: Path) -> object:
    """Load a skops file, rejecting any type outside the allow-list."""
    if not path.is_file():
        raise ArtifactNotFoundError(
            f"baseline artifact not found at {path}; run `stc train baseline`"
        )
    try:
        untrusted = skops_io.get_untrusted_types(file=path)
    except Exception as error:
        raise IntegrityError(f"{path.name} is not a readable skops file") from error
    rejected = [name for name in untrusted if not is_allowed_type(name)]
    if rejected:
        raise UntrustedArtifactError(
            "refusing to load untrusted skops type(s): " + ", ".join(sorted(rejected))
        )
    return skops_io.load(path, trusted=untrusted)


def cast_linear_coefficients(pipeline: object) -> bool:
    """Cast fitted linear coefficients to float32 in place; report whether it happened."""
    steps = getattr(pipeline, "named_steps", None)
    if not isinstance(steps, Mapping):
        return False
    estimator = steps.get("classifier")
    if estimator is None:
        return False
    changed = False
    for name in ("coef_", "intercept_"):
        value = getattr(estimator, name, None)
        if value is None:
            continue
        array: NDArray[np.float64] = np.asarray(value)
        if array.dtype != np.float32:
            setattr(estimator, name, array.astype(np.float32))
            changed = True
    return changed


def _read_json(path: Path, *, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise ArtifactNotFoundError(f"{description} not found at {path}; run `stc train baseline`")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{description} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise IntegrityError(f"{description} must be a JSON object")
    return payload


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


class SkopsBaselineStore:
    """Persist ``baseline.skops``, ``labels.json`` and a hashed ``manifest.json``."""

    def __init__(self, max_bytes: int = MAX_BASELINE_BYTES) -> None:
        self._max_bytes = max_bytes

    def save(
        self,
        classifier: Classifier,
        dest: Path,
        *,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Write the baseline directory and return its manifest."""
        pipeline = getattr(classifier, "pipeline", None)
        if pipeline is None:
            raise InvalidInputError("the baseline store needs a classifier with a .pipeline")
        target = Path(dest)
        target.mkdir(parents=True, exist_ok=True)
        skops_path = target / "baseline.skops"
        labels_path = target / "labels.json"

        save(pipeline, skops_path)
        float32_cast = False
        if skops_path.stat().st_size > self._max_bytes:
            float32_cast = cast_linear_coefficients(pipeline)
            save(pipeline, skops_path)

        labels = classifier.labels
        labels_payload = {"schema_version": 1, "intents": list(labels.intents)}
        labels_path.write_text(
            json.dumps(labels_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        files = [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in (skops_path, labels_path)
        ]
        winner = metadata.get("winner", {})
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": MANIFEST_KIND,
            "name": classifier.info.model_id,
            "version": classifier.info.version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stc_version": __version__,
            "model": {"parameters": classifier.info.parameters},
            "labels": {
                "n_intents": labels.n_intents,
                "n_scenarios": labels.n_scenarios,
            },
            "winner": dict(_mapping(winner, "winner")),
            "grid": metadata.get("grid", {}),
            "dataset": metadata.get("dataset", {}),
            "quick": bool(metadata.get("quick", False)),
            "n_configs": metadata.get("n_configs", 0),
            "float32_cast": float32_cast,
            "files": files,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest

    def load(self, source: Path) -> Classifier:
        """Verify the manifest, then rebuild the classifier around the pipeline."""
        root = Path(source)
        if not root.is_dir():
            raise ArtifactNotFoundError(
                f"baseline directory not found at {root}; run `stc train baseline`"
            )
        manifest = _read_json(root / "manifest.json", description="baseline manifest")
        if manifest.get("kind") != MANIFEST_KIND:
            raise IntegrityError(f"expected kind {MANIFEST_KIND!r}, found {manifest.get('kind')!r}")
        for entry in cast(list[object], manifest.get("files", [])):
            item = _mapping(entry, "manifest files entry")
            relative = item.get("path")
            expected = item.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise IntegrityError("manifest files entries need path and sha256")
            artifact = root / relative
            if not artifact.is_file():
                raise IntegrityError(f"manifest lists a missing file: {relative}")
            if sha256_file(artifact) != expected:
                raise IntegrityError(f"checksum mismatch for {relative}")

        labels_payload = _read_json(root / "labels.json", description="baseline labels")
        raw_intents = labels_payload.get("intents")
        if not isinstance(raw_intents, list) or not all(
            isinstance(intent, str) for intent in raw_intents
        ):
            raise IntegrityError("labels.json must contain a list of intent strings")
        labels = LabelSpace.from_intents(cast(list[str], raw_intents))
        pipeline = load(root / "baseline.skops")

        from stc.adapters.models.tfidf_linear import TfidfLinearClassifier

        model_section = _mapping(manifest.get("model", {}), "model")
        parameters = model_section.get("parameters")
        info = ModelInfo(
            model_id=str(manifest.get("name", "tfidf-linear-baseline")),
            kind=MANIFEST_KIND,
            version=str(manifest.get("version", __version__)),
            quantization=None,
            parameters=parameters if isinstance(parameters, int) else None,
            license="MIT",
            n_intents=labels.n_intents,
            n_scenarios=labels.n_scenarios,
        )
        return TfidfLinearClassifier(pipeline=pipeline, labels=labels, info=info)
