"""Explicit domain errors mapped to stable CLI exit codes."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every expected failure of the domain and application layers."""

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(DomainError):
    """A command or API call received invalid input."""

    exit_code = 2


class LabelSpaceError(DomainError):
    """An unknown or malformed label was used."""

    exit_code = 2


class MissingExtraError(DomainError):
    """A feature needs an optional dependency that is not installed."""

    exit_code = 2


class ChecksumMismatchError(DomainError):
    """A downloaded artifact does not match its pinned checksum."""

    exit_code = 3


class IntegrityError(DomainError):
    """An artifact, bundle or manifest failed its integrity checks."""

    exit_code = 3


class UntrustedArtifactError(DomainError):
    """An artifact contains types or paths that are not allow-listed."""

    exit_code = 3


class ArtifactNotFoundError(DomainError):
    """A model, bundle or dataset could not be located."""

    exit_code = 4


class OfflineError(DomainError):
    """A network operation was refused because the tool runs offline."""

    exit_code = 5


class TestSplitLocked(DomainError):
    """The test split was read without the final-evaluation capability token."""

    exit_code = 6
