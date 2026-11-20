"""Tests for the test-split capability token and guard."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from stc.domain.access import (
    TestAccessEntry as AccessEntry,
)
from stc.domain.access import (
    TestAccessGrant as AccessGrant,
)
from stc.domain.access import (
    TestSplitGuard,
    require_test_grant,
)
from stc.domain.errors import TestSplitLocked as SplitLockedError
from stc.domain.splits import Split

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class FixedClock:
    now_time: datetime

    def now(self) -> datetime:
        return self.now_time


def test_guard_issues_grant_from_clock() -> None:
    grant = TestSplitGuard().grant("final-report", "stc-es-intent", FixedClock(NOW))
    assert grant == AccessGrant(
        purpose="final-report",
        model_id="stc-es-intent",
        issued_at=NOW,
    )


def test_guard_records_the_scored_model_sha256() -> None:
    grant = TestSplitGuard().grant(
        "final-report",
        "stc-es-intent",
        FixedClock(NOW),
        model_sha256="0123456789abcdef",
    )
    assert grant.model_sha256 == "0123456789abcdef"


def test_test_split_without_grant_is_locked() -> None:
    with pytest.raises(SplitLockedError) as excinfo:
        require_test_grant(Split.TEST, None)
    assert excinfo.value.exit_code == 6
    assert "locked" in str(excinfo.value)
    assert "--final" in str(excinfo.value)


def test_test_split_with_grant_passes() -> None:
    grant = TestSplitGuard().grant("final-report", "stc-es-intent", FixedClock(NOW))
    require_test_grant(Split.TEST, grant)


@pytest.mark.parametrize("split", [Split.TRAIN, Split.DEV])
def test_other_splits_never_need_a_grant(split: Split) -> None:
    require_test_grant(split, None)


def test_access_entry_carries_grant_and_audit_fields() -> None:
    grant = TestSplitGuard().grant("final-report", "stc-es-intent", FixedClock(NOW))
    entry = AccessEntry(
        grant=grant,
        command="stc report final",
        git_sha="abc1234",
        n_examples=2974,
    )
    assert entry.grant.purpose == "final-report"
    assert entry.command == "stc report final"
    assert entry.git_sha == "abc1234"
    assert entry.n_examples == 2974
