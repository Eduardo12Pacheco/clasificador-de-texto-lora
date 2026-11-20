"""Request IDs, security headers and access logging for the API (D14/S14)."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from stc.logging import get_logger

__all__ = ["SECURITY_HEADERS", "RequestContextMiddleware"]

REQUEST_ID_HEADER = "X-Request-ID"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _request_id(candidate: str | None) -> str:
    if candidate is not None and 0 < len(candidate) <= 100 and candidate.isprintable():
        return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, security headers and one structured access log line."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = _request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        request.state.n_texts = None
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        response.headers[REQUEST_ID_HEADER] = request_id
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        get_logger("stc.api").info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=elapsed_ms,
            n_texts=getattr(request.state, "n_texts", None),
        )
        return response
