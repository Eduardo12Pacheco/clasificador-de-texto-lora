#!/usr/bin/env python3
"""Capture the real UI screenshots used by the README (T10/T15)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import Page, Route, sync_playwright

from ui_harness import DEFAULT_BUNDLE, seed_preferences, serve_ui

VIEWPORTS = {"desktop": {"width": 1280, "height": 900}, "mobile": {"width": 360, "height": 740}}


def _wait_for_model(page: Page) -> None:
    page.wait_for_function(
        "() => !document.querySelector('#model-chip').textContent.includes('Cargando')"
    )


def _open(page: Page, base_url: str, wait_for_model: bool = True) -> None:
    page.goto(base_url + "/", wait_until="networkidle")
    if wait_for_model:
        _wait_for_model(page)


def _show_result(page: Page) -> None:
    page.locator(".example-chip").first.click()
    page.locator("#classify-button").click()
    page.wait_for_selector('#result[data-state="ready"]', timeout=15_000)


def _problem_500(route: Route) -> None:
    route.fulfill(
        status=500,
        content_type="application/problem+json",
        body=json.dumps(
            {
                "type": "about:blank",
                "title": "internal server error",
                "status": 500,
                "detail": "synthetic failure used for the error-state screenshot",
            }
        ),
    )


def capture(out_dir: Path, bundle: Path) -> list[Path]:
    """Start the real app and write the documented screenshot set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with serve_ui(bundle) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        desktop = browser.new_context(viewport=VIEWPORTS["desktop"], locale="es-ES")
        page = desktop.new_page()
        seed_preferences(page, theme="light")
        _open(page, base_url)
        _show_result(page)
        light = out_dir / "ui-light.png"
        page.screenshot(path=str(light), full_page=True)
        written.append(light)

        page.locator("#theme-toggle").click()
        page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
        dark = out_dir / "ui-dark.png"
        page.screenshot(path=str(dark), full_page=True)
        written.append(dark)

        if page.locator("#compare-control").is_visible():
            page.locator("#compare-toggle").check()
            page.wait_for_selector("#compare:not([hidden])", timeout=15_000)
            compare = out_dir / "ui-compare.png"
            page.screenshot(path=str(compare), full_page=True)
            written.append(compare)

        error_page = desktop.new_page()
        seed_preferences(error_page, theme="light")
        error_page.route("**/v1/predict", _problem_500)
        _open(error_page, base_url)
        error_page.locator(".example-chip").first.click()
        error_page.locator("#classify-button").click()
        error_page.wait_for_selector('#result[data-state="error"]', timeout=15_000)
        error = out_dir / "ui-error.png"
        error_page.screenshot(path=str(error), full_page=True)
        written.append(error)

        mobile = browser.new_context(viewport=VIEWPORTS["mobile"], locale="es-ES")
        mobile_page = mobile.new_page()
        seed_preferences(mobile_page, theme="light")
        _open(mobile_page, base_url)
        _show_result(mobile_page)
        mobile_shot = out_dir / "ui-mobile.png"
        mobile_page.screenshot(path=str(mobile_shot), full_page=True)
        written.append(mobile_shot)

        for context in (desktop, mobile):
            context.close()
        browser.close()

    for path in written:
        print(f"{path.name}: {path.stat().st_size} bytes")
    return written


def main() -> int:
    """Parse arguments and capture the screenshots."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Directory for the PNG files.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Model bundle to serve (default: the committed tiny test bundle).",
    )
    arguments = parser.parse_args()
    written = capture(arguments.out.resolve(), arguments.bundle.resolve())
    if len(written) < 4:
        raise SystemExit("expected at least the four baseline screenshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
