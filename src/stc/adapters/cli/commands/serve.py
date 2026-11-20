"""``stc serve``: run the local API and web UI with uvicorn (D14)."""

from __future__ import annotations

import ipaddress
from typing import Annotated

import typer
import uvicorn

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.logging import configure_logging, get_logger

__all__ = ["serve"]


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@handle_domain_errors
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Interface to bind (default: STC_API_HOST)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Port to bind (default: STC_API_PORT)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name, path or 'baseline' (default: STC_MODEL)."),
    ] = None,
) -> None:
    """Serve the FastAPI app and the bundled UI on the local machine."""
    settings = bootstrap.build_settings()
    resolved_host = host if host is not None else settings.api_host
    resolved_port = port if port is not None else settings.api_port
    if not _is_loopback(resolved_host):
        message = f"warning: binding {resolved_host}; the API is intended for loopback use only"
        typer.echo(message, err=True)
        get_logger("stc.cli").warning("non_loopback_bind", host=resolved_host)
    configure_logging(settings.log_level, settings.resolve_log_format())
    application = bootstrap.build_api_app(settings, model_selector=model)
    uvicorn.run(application, host=resolved_host, port=resolved_port, log_level="info")
