"""Append-only JSONL audit trail for authorised test-split reads (S3)."""

from __future__ import annotations

import json
from pathlib import Path

from stc.domain.access import TestAccessEntry

__all__ = ["JsonlAccessLog"]


class JsonlAccessLog:
    """Append one deterministic JSON line per audited test-split read."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """The file this log appends to."""
        return self._path

    def record(self, entry: TestAccessEntry) -> None:
        """Append ``entry`` without rewriting earlier lines."""
        payload = {
            "command": entry.command,
            "git_sha": entry.git_sha,
            "model_id": entry.grant.model_id,
            "n_examples": entry.n_examples,
            "purpose": entry.grant.purpose,
            "timestamp": entry.grant.issued_at.isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
