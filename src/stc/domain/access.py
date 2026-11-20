"""Capability-token guard that keeps the test split unreadable by accident."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from stc.domain.errors import TestSplitLocked
from stc.domain.splits import Split

_LOCKED_MESSAGE = "test split is locked; pass --final to record an audited evaluation"


class Clock(Protocol):
    """Minimal clock port so token issuance is deterministic in tests."""

    def now(self) -> datetime:
        """Return the current UTC timestamp."""
        ...


@dataclass(frozen=True, slots=True)
class TestAccessGrant:
    """Capability token authorising one audited read of the test split."""

    purpose: str
    model_id: str
    issued_at: datetime


@dataclass(frozen=True, slots=True)
class TestAccessEntry:
    """Audit-log record written for every authorised test read."""

    grant: TestAccessGrant
    command: str
    git_sha: str
    n_examples: int


class TestSplitGuard:
    """Issue test-access grants to the few explicitly audited use cases."""

    def grant(self, purpose: str, model_id: str, clock: Clock) -> TestAccessGrant:
        """Issue a grant stamped with ``clock.now()``."""
        return TestAccessGrant(
            purpose=purpose,
            model_id=model_id,
            issued_at=clock.now(),
        )


def require_test_grant(split: Split, grant: TestAccessGrant | None) -> None:
    """Refuse to read the test split unless a capability token is present."""
    if split is Split.TEST and grant is None:
        raise TestSplitLocked(_LOCKED_MESSAGE)
