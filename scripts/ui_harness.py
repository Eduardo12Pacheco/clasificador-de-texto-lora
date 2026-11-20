"""Shared uvicorn/Playwright harness for the UI capture and GIF scripts."""

from __future__ import annotations

import contextlib
import json
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import uvicorn

from stc.adapters.api.app import create_app
from stc.adapters.api.provider import BootstrapClassifierProvider
from stc.config import Settings

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny_bundle"


@contextlib.contextmanager
def serve_ui(bundle: Path = DEFAULT_BUNDLE) -> Iterator[str]:
    """Serve the real app on an ephemeral loopback port and yield its base URL."""
    resolved_bundle = bundle.resolve()
    with tempfile.TemporaryDirectory(prefix="stc-ui-home-") as home:
        settings = Settings(home=Path(home), model=str(resolved_bundle))
        provider = BootstrapClassifierProvider(settings, default_selector=str(resolved_bundle))
        application = create_app(settings, provider)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = uvicorn.Server(
            uvicorn.Config(application, host="127.0.0.1", port=port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        # The resolver looks for ``baseline`` relative to the working directory; a bundle
        # that carries ``baseline/`` is therefore served from inside the bundle.
        workdir = resolved_bundle if (resolved_bundle / "baseline").is_dir() else Path(home)
        with contextlib.chdir(workdir):
            thread.start()
            for _ in range(200):
                if server.started:
                    break
                if not thread.is_alive():
                    raise RuntimeError("the UI server stopped before it was ready")
                time.sleep(0.025)
            if not server.started:
                server.should_exit = True
                thread.join(timeout=5)
                raise RuntimeError("the UI server did not start within 5 seconds")
            try:
                yield f"http://127.0.0.1:{port}"
            finally:
                server.should_exit = True
                thread.join(timeout=10)


def seed_preferences(page: object, *, language: str = "es", theme: str = "light") -> None:
    """Persist preferences before the page module runs (Playwright ``Page``)."""
    language_literal = json.dumps(language)
    theme_literal = json.dumps(theme)
    page.add_init_script(  # type: ignore[attr-defined]
        "try { if (!sessionStorage.getItem('stc.seeded')) {"
        " sessionStorage.setItem('stc.seeded', '1');"
        f" localStorage.setItem('stc.lang', {language_literal});"
        f" localStorage.setItem('stc.theme', {theme_literal}); }} }}"
        " catch (error) { /* storage may be blocked */ }"
    )
