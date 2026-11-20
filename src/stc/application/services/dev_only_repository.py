"""Decorator that removes test-split access from every non-final use case."""

from __future__ import annotations

from collections.abc import Sequence

from stc.application.dto import DatasetMeta
from stc.application.ports import DatasetRepository
from stc.domain.access import TestAccessGrant, require_test_grant
from stc.domain.examples import Example
from stc.domain.splits import Split


class DevOnlyRepository:
    """Wrap a repository so that only ``load_test`` can read the test split."""

    def __init__(self, inner: DatasetRepository) -> None:
        self._inner = inner

    def load(self, split: Split) -> Sequence[Example]:
        """Read a split; the test split always raises ``TestSplitLocked``."""
        require_test_grant(split, None)
        return self._inner.load(split)

    def load_test(self, grant: TestAccessGrant | None) -> Sequence[Example]:
        """Read the test split with an audited capability token."""
        require_test_grant(Split.TEST, grant)
        return self._inner.load(Split.TEST)

    def save(self, split: Split, examples: Sequence[Example]) -> None:
        """Persist a split through to the wrapped repository."""
        self._inner.save(split, examples)

    def meta(self) -> DatasetMeta:
        """Read provenance metadata from the wrapped repository."""
        return self._inner.meta()

    def save_meta(self, meta: DatasetMeta) -> None:
        """Persist provenance metadata through the wrapped repository."""
        self._inner.save_meta(meta)
