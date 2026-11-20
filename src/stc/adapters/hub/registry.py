"""Committed model registry metadata (D13).

The registry never contains weights: it maps a model name/version to the
release URL, size and sha256 that ``stc model pull`` must verify.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from stc.domain.errors import ArtifactNotFoundError, IntegrityError

__all__ = ["REGISTRY_VERSION", "ModelRegistry", "RegistryEntry"]

REGISTRY_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _string(payload: object, field_name: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise IntegrityError(f"registry {field_name} must be a non-empty string")
    return payload


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One published model bundle."""

    name: str
    version: str
    url: str
    sha256: str
    size_bytes: int
    license: str

    def __post_init__(self) -> None:
        if not _SHA256.match(self.sha256):
            raise IntegrityError("registry sha256 must be 64 lowercase hex characters")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 1:
            raise IntegrityError("registry size_bytes must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation with stable keys."""
        return {
            "license": self.license,
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "url": self.url,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, payload: object) -> RegistryEntry:
        """Validate one registry row."""
        if not isinstance(payload, dict):
            raise IntegrityError("registry models entries must be objects")
        row = cast(dict[str, object], payload)
        size = row.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int):
            raise IntegrityError("registry size_bytes must be an integer")
        return cls(
            name=_string(row.get("name"), "name"),
            version=_string(row.get("version"), "version"),
            url=_string(row.get("url"), "url"),
            sha256=_string(row.get("sha256"), "sha256"),
            size_bytes=size,
            license=_string(row.get("license"), "license"),
        )


class ModelRegistry:
    """An ordered, validated set of published model entries."""

    def __init__(self, entries: list[RegistryEntry] | None = None) -> None:
        self._entries = list(entries or [])

    @property
    def entries(self) -> tuple[RegistryEntry, ...]:
        """Every registered entry in insertion order."""
        return tuple(self._entries)

    @classmethod
    def empty(cls) -> ModelRegistry:
        """Return the empty registry committed with the source tree."""
        return cls([])

    @classmethod
    def from_file(cls, path: Path) -> ModelRegistry:
        """Read and validate ``model_registry.json``."""
        source = Path(path)
        if not source.is_file():
            raise ArtifactNotFoundError(f"model registry not found at {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IntegrityError("model registry is not valid JSON") from error
        if not isinstance(payload, dict):
            raise IntegrityError("model registry must be a JSON object")
        if payload.get("registry_version") != REGISTRY_VERSION:
            raise IntegrityError(
                f"unsupported registry_version {payload.get('registry_version')!r}"
            )
        models = payload.get("models")
        if not isinstance(models, list):
            raise IntegrityError("registry models must be a list")
        return cls([RegistryEntry.from_mapping(entry) for entry in models])

    def find(self, name: str, version: str | None = None) -> RegistryEntry:
        """Return one entry by name (optionally pinned to a version)."""
        for entry in self._entries:
            if entry.name == name and (version is None or entry.version == version):
                return entry
        suffix = f" {version}" if version else ""
        raise ArtifactNotFoundError(
            f"model {name}{suffix} is not in the registry; run: stc model list"
        )

    def upsert(self, entry: RegistryEntry) -> ModelRegistry:
        """Return a new registry with ``entry`` inserted or replaced."""
        rows = [
            existing
            for existing in self._entries
            if not (existing.name == entry.name and existing.version == entry.version)
        ]
        rows.append(entry)
        return ModelRegistry(rows)

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic JSON payload."""
        return {
            "models": [entry.to_dict() for entry in self._entries],
            "registry_version": REGISTRY_VERSION,
        }

    def save(self, path: Path) -> None:
        """Write the registry deterministically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
