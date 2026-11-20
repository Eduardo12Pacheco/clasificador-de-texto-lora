"""Strategy factory that picks a ``Classifier`` adapter from ``manifest.kind``.

The factory is pure: the composition root registers one builder per adapter
kind, so the application layer never imports an adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from stc.application.ports import Classifier
from stc.domain.errors import InvalidInputError

__all__ = ["ClassifierFactory"]

ClassifierBuilder = Callable[[Path], Classifier]


class ClassifierFactory:
    """Create classifiers by artifact kind through injected builders."""

    def __init__(self, builders: Mapping[str, ClassifierBuilder]) -> None:
        self._builders = dict(builders)

    @property
    def kinds(self) -> tuple[str, ...]:
        """Every artifact kind the factory can build."""
        return tuple(sorted(self._builders))

    def create(self, kind: str, path: Path) -> Classifier:
        """Build a classifier for ``kind`` from ``path``."""
        builder = self._builders.get(kind)
        if builder is None:
            raise InvalidInputError(
                f"unsupported model kind {kind!r}; expected one of {', '.join(self.kinds)}"
            )
        return builder(Path(path))
