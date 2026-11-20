"""Tests for the domain error hierarchy and its CLI exit codes."""

import pytest

from stc.domain import errors

pytestmark = pytest.mark.unit

ERRORS_WITH_EXIT_CODE = [
    (errors.ChecksumMismatchError, 3),
    (errors.IntegrityError, 3),
    (errors.UntrustedArtifactError, 3),
    (errors.ArtifactNotFoundError, 4),
    (errors.OfflineError, 5),
    (errors.TestSplitLocked, 6),
    (errors.InvalidInputError, 2),
    (errors.LabelSpaceError, 2),
    (errors.MissingExtraError, 2),
]


@pytest.mark.parametrize(("error", "exit_code"), ERRORS_WITH_EXIT_CODE)
def test_domain_error_exit_codes(error: type[errors.DomainError], exit_code: int) -> None:
    assert issubclass(error, errors.DomainError)
    assert error.exit_code == exit_code


def test_domain_error_carries_its_message() -> None:
    error = errors.LabelSpaceError("unknown intent: nope")
    assert error.message == "unknown intent: nope"
    assert str(error) == "unknown intent: nope"


def test_domain_error_is_catchable_by_base_class() -> None:
    with pytest.raises(errors.DomainError):
        raise errors.OfflineError("network disabled")


def test_base_domain_error_uses_exit_code_one() -> None:
    assert errors.DomainError.exit_code == 1
