"""RFC 7807 problem responses for every API error path (S14/D16)."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from stc.adapters.api.schemas import ProblemOut
from stc.domain.errors import (
    ArtifactNotFoundError,
    ChecksumMismatchError,
    DomainError,
    IntegrityError,
    InvalidInputError,
    LabelSpaceError,
    MissingExtraError,
    OfflineError,
    TestSplitLocked,
    UntrustedArtifactError,
)

__all__ = ["ProblemError", "install_error_handlers", "problem_response"]

PROBLEM_MEDIA_TYPE = "application/problem+json"


class ProblemError(Exception):
    """An HTTP problem that is not caused by a domain error (e.g. missing baseline)."""

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail


def problem_response(status: int, title: str, detail: str) -> JSONResponse:
    """Build an ``application/problem+json`` response."""
    payload = ProblemOut(
        type="about:blank",
        title=title,
        status=status,
        detail=detail,
    )
    return JSONResponse(
        status_code=status,
        content=payload.model_dump(),
        media_type=PROBLEM_MEDIA_TYPE,
    )


def _status_for(error: DomainError) -> tuple[int, str]:
    if isinstance(error, InvalidInputError | LabelSpaceError):
        return 422, "invalid request"
    if isinstance(error, ArtifactNotFoundError):
        return 503, "model unavailable"
    if isinstance(error, OfflineError):
        return 503, "offline"
    if isinstance(error, IntegrityError | ChecksumMismatchError | UntrustedArtifactError):
        return 503, "unsafe artifact"
    if isinstance(error, MissingExtraError):
        return 500, "missing optional dependency"
    if isinstance(error, TestSplitLocked):
        return 403, "test split locked"
    return 500, "internal server error"


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    del request
    if not isinstance(exc, DomainError):
        return problem_response(500, "internal server error", "unexpected error")
    status, title = _status_for(exc)
    detail = exc.message
    if isinstance(exc, ArtifactNotFoundError) and "stc model pull" not in detail:
        detail = f"{detail}; run: stc model pull"
    return problem_response(status, title, detail)


async def _problem_error_handler(request: Request, exc: Exception) -> JSONResponse:
    del request
    if not isinstance(exc, ProblemError):
        return problem_response(500, "internal server error", "unexpected error")
    return problem_response(exc.status, exc.title, exc.detail)


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    del request
    if not isinstance(exc, RequestValidationError):
        return problem_response(422, "invalid request", "request validation failed")
    locations = ", ".join(
        ".".join(str(part) for part in error.get("loc", ())) for error in exc.errors()
    )
    detail = "request validation failed"
    if locations:
        detail = f"{detail}: {locations}"
    return problem_response(422, "invalid request", detail)


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    del request, exc
    return problem_response(500, "internal server error", "internal server error")


def install_error_handlers(application: FastAPI) -> None:
    """Register the handlers on the FastAPI application."""
    application.add_exception_handler(DomainError, _domain_error_handler)
    application.add_exception_handler(ProblemError, _problem_error_handler)
    application.add_exception_handler(RequestValidationError, _validation_error_handler)
    application.add_exception_handler(Exception, _unhandled_error_handler)
