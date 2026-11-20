"""End-to-end browser tests for the bundled web UI (S15)."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, expect, sync_playwright

from stc.adapters.api.app import create_app
from stc.adapters.api.provider import BootstrapClassifierProvider
from stc.adapters.inference.bundle_io import refresh_manifest
from stc.config import Settings

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
MOBILE_VIEWPORT = {"width": 360, "height": 740}
DESKTOP_VIEWPORT = {"width": 1280, "height": 900}


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture(scope="session")
def bundle_with_baseline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("e2e-baseline")
    bundle = root / "bundle"
    shutil.copytree(TINY_BUNDLE, bundle)
    environment = {
        **os.environ,
        "STC_HOME": str(root / "home"),
        "STC_MODEL": str(bundle),
        "STC_LOG_LEVEL": "WARNING",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stc",
            "train",
            "baseline",
            "--dataset",
            "fixture",
            "--quick",
            "--out",
            str(bundle / "baseline"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    # ``stc bundle pack`` re-hashes every file; do the same after adding baseline/.
    refresh_manifest(bundle)
    return bundle


@contextlib.contextmanager
def _serve(bundle: Path, home: Path) -> Iterator[str]:
    settings = Settings(home=home, model=str(bundle))
    provider = BootstrapClassifierProvider(settings, default_selector=str(bundle))
    application = create_app(settings, provider)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(
        uvicorn.Config(application, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    home.mkdir(parents=True, exist_ok=True)
    workdir = bundle if (bundle / "baseline").is_dir() else home
    with contextlib.chdir(workdir):
        thread.start()
        for _ in range(200):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError("the e2e server stopped before it was ready")
            time.sleep(0.025)
        if not server.started:
            raise RuntimeError("the e2e server did not start")
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)


@pytest.fixture
def app_url(bundle_with_baseline: Path, tmp_path: Path) -> Iterator[str]:
    with _serve(bundle_with_baseline, tmp_path / "home") as url:
        yield url


@pytest.fixture
def bare_app_url(tmp_path: Path) -> Iterator[str]:
    with _serve(TINY_BUNDLE, tmp_path / "home") as url:
        yield url


@contextlib.contextmanager
def _open_page(
    browser: Browser,
    app_url: str,
    *,
    viewport: dict[str, int] | None = None,
    theme: str = "light",
) -> Iterator[Page]:
    context = browser.new_context(
        viewport=viewport or DESKTOP_VIEWPORT,
        locale="es-ES",
        color_scheme="light",
    )
    page = context.new_page()
    page.add_init_script(
        "try { if (!sessionStorage.getItem('stc.seeded')) {"
        " sessionStorage.setItem('stc.seeded', '1');"
        " localStorage.setItem('stc.lang', 'es');"
        f" localStorage.setItem('stc.theme', '{theme}'); }} }}"
        " catch (error) { /* storage may be blocked */ }"
    )
    page.goto(app_url + "/", wait_until="networkidle")
    expect(page.locator("#model-chip")).not_to_have_text("Cargando modelo…")
    try:
        yield page
    finally:
        context.close()


def _classify(page: Page, text: str) -> None:
    page.locator("#text-input").fill(text)
    page.locator("#classify-button").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "ready", timeout=15_000)


def test_classify_flow_matches_the_api_response(browser: Browser, app_url: str) -> None:
    text = "despiértame a las seis de la mañana"
    with _open_page(browser, app_url) as page:
        expect(page.locator("#model-chip")).to_contain_text("stc-tiny-test")
        _classify(page, text)
        api = page.request.post(
            app_url + "/v1/predict",
            data=json.dumps({"texts": [text], "top_k": 3}),
            headers={"content-type": "application/json"},
        )
        assert api.ok
        payload = api.json()

        expect(page.locator("#top-intent")).to_have_text(payload["predictions"][0]["intent"])
        expect(page.locator("#top-k .candidate")).to_have_count(3)
        expect(page.locator("#confidence-value")).not_to_have_text("—")
        expect(page.locator("#latency-chip")).to_contain_text("ms")


def test_keyboard_shortcut_submits_the_form(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        page.locator("#text-input").fill("apaga la luz")
        page.locator("#text-input").press("Control+Enter")
        expect(page.locator("#result")).to_have_attribute("data-state", "ready", timeout=15_000)
        expect(page.locator("#top-intent")).not_to_have_text("—")


def test_example_chip_fills_the_textarea(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        chip = page.locator(".example-chip").nth(2)
        text = chip.get_attribute("data-example")
        chip.click()
        expect(page.locator("#text-input")).to_have_value(text or "")
        expect(page.locator("#input-counter")).to_have_text(f"{len(text or '')}/512")


def test_error_alert_retries_successfully(browser: Browser, app_url: str) -> None:
    calls = {"count": 0}

    def fail_once(route: object) -> None:
        if calls["count"] == 0:
            calls["count"] = 1
            route.fulfill(  # type: ignore[attr-defined]
                status=500,
                content_type="application/problem+json",
                body=json.dumps(
                    {
                        "type": "about:blank",
                        "title": "internal server error",
                        "status": 500,
                        "detail": "synthetic test failure",
                    }
                ),
            )
        else:
            route.continue_()  # type: ignore[attr-defined]

    with _open_page(browser, app_url) as page:
        page.route("**/v1/predict", fail_once)
        page.locator(".example-chip").first.click()
        page.locator("#classify-button").click()
        expect(page.locator("#result-error")).to_be_visible(timeout=15_000)
        expect(page.locator("#error-detail")).to_contain_text("synthetic test failure")
        page.locator("#retry-button").click()
        expect(page.locator("#result")).to_have_attribute("data-state", "ready", timeout=15_000)


def test_offline_banner_follows_browser_connectivity(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        expect(page.locator("#offline-banner")).to_be_hidden()
        page.context.set_offline(True)
        expect(page.locator("#offline-banner")).to_be_visible()
        page.context.set_offline(False)
        expect(page.locator("#offline-banner")).to_be_hidden()


def test_dark_theme_toggle_persists_after_reload(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url, theme="light") as page:
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        page.reload(wait_until="networkidle")
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")


def test_language_toggle_changes_strings_and_persists(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        expect(page.locator("h1")).to_have_text("Clasificador de intenciones")
        page.locator("#lang-toggle").click()
        expect(page.locator("h1")).to_have_text("Intent classifier")
        expect(page.locator("html")).to_have_attribute("lang", "en")
        page.reload(wait_until="networkidle")
        expect(page.locator("h1")).to_have_text("Intent classifier")


def test_mobile_viewport_has_no_horizontal_scroll(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url, viewport=MOBILE_VIEWPORT) as page:
        _classify(page, "¿cómo está el clima en Quito?")
        light_fits = page.evaluate(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
        page.locator("#theme-toggle").click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        dark_fits = page.evaluate(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
        assert light_fits is True
        assert dark_fits is True


def test_compare_toggle_is_hidden_without_a_baseline(browser: Browser, bare_app_url: str) -> None:
    with _open_page(browser, bare_app_url) as page:
        expect(page.locator("#compare-control")).to_be_hidden()
        page.locator("#model-panel summary").click()
        expect(page.locator("#model-baseline")).to_have_text("No")
        _classify(page, "ponme música de Juanes")
        expect(page.locator("#compare")).to_be_hidden()


def test_compare_mode_shows_both_models(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        _classify(page, "envía un correo a Pedro")
        expect(page.locator("#compare-control")).to_be_visible()
        page.locator("#compare-toggle").check()
        expect(page.locator("#compare")).to_be_visible(timeout=15_000)
        expect(page.locator("#compare-default-intent")).not_to_have_text("—")
        expect(page.locator("#compare-baseline-intent")).not_to_have_text("—")
        agreement = page.locator("#compare-agreement").get_attribute("data-state")
        assert agreement in {"agree", "disagree"}


def test_requests_stay_same_origin_and_csp_is_present(browser: Browser, app_url: str) -> None:
    requests: list[str] = []
    with _open_page(browser, app_url) as page:
        page.on("request", lambda request: requests.append(request.url))
        response = page.reload(wait_until="networkidle")
        assert response is not None
        assert response.headers.get("content-security-policy") == "default-src 'self'"
        _classify(page, "pide una pizza grande")

    assert requests
    assert all(url.startswith(app_url) for url in requests), requests


def test_model_panel_reports_calibration_metadata(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        page.locator("#model-panel summary").click()
        expect(page.locator("#model-id")).to_have_text("stc-tiny-test")
        expect(page.locator("#model-version")).to_have_text("0.0.1")
        expect(page.locator("#model-quantization")).to_have_text("fp32")
        expect(page.locator("#model-temperature")).to_have_text("1,50")
        expect(page.locator("#model-threshold")).to_have_text("0,50")
        expect(page.locator("#model-baseline")).to_have_text("Sí")


def test_model_not_installed_error_explains_the_fix(browser: Browser, app_url: str) -> None:
    def missing_model(route: object) -> None:
        route.fulfill(  # type: ignore[attr-defined]
            status=503,
            content_type="application/problem+json",
            body=json.dumps(
                {
                    "type": "about:blank",
                    "title": "model unavailable",
                    "status": 503,
                    "detail": "model not found",
                }
            ),
        )

    with _open_page(browser, app_url) as page:
        page.route("**/v1/predict", missing_model)
        page.locator(".example-chip").first.click()
        page.locator("#classify-button").click()
        expect(page.locator("#result-error")).to_be_visible(timeout=15_000)
        expect(page.locator("#error-hint")).to_have_text("stc model pull")


def test_clear_button_resets_the_result(browser: Browser, app_url: str) -> None:
    with _open_page(browser, app_url) as page:
        _classify(page, "cuéntame un chiste")
        page.locator("#clear-button").click()
        expect(page.locator("#text-input")).to_have_value("")
        expect(page.locator("#result")).to_have_attribute("data-state", "empty")
        expect(page.locator("#classify-button")).to_be_disabled()
