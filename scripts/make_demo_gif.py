#!/usr/bin/env python3
"""Record a short real UI interaction and assemble the README demo GIF."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

from ui_harness import DEFAULT_BUNDLE, seed_preferences, serve_ui

MAX_GIF_BYTES = 3_000_000


def _snap(page: Page, frames: list[Image.Image], seconds: float, fps: int) -> None:
    interval = 1.0 / fps
    deadline = time.monotonic() + seconds
    next_frame = time.monotonic()
    while time.monotonic() < deadline:
        frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
        next_frame += interval
        delay = next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _encode(frames: list[Image.Image], *, width: int, fps: int, colors: int) -> bytes:
    resized: list[Image.Image] = []
    for frame in frames:
        height = round(frame.height * width / frame.width)
        resized.append(frame.resize((width, height), Image.Resampling.LANCZOS))
    paletted = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors) for frame in resized
    ]
    buffer = io.BytesIO()
    paletted[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=paletted[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
        disposal=1,
    )
    return buffer.getvalue()


def record(out_path: Path, bundle: Path, *, width: int, fps: int, colors: int) -> Path:
    """Record the interaction at the real app and write a bounded GIF."""
    frames: list[Image.Image] = []
    with serve_ui(bundle) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1280, "height": 800}, locale="es-ES", color_scheme="light"
        )
        page = context.new_page()
        seed_preferences(page, theme="light")
        page.goto(base_url + "/", wait_until="networkidle")
        page.wait_for_function(
            "() => !document.querySelector('#model-chip').textContent.includes('Cargando')"
        )
        _snap(page, frames, 1.5, fps)

        page.locator(".example-chip").first.click()
        _snap(page, frames, 1.2, fps)

        page.locator("#classify-button").click()
        page.wait_for_selector('#result[data-state="ready"]', timeout=15_000)
        _snap(page, frames, 3.0, fps)

        page.locator("#theme-toggle").click()
        page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
        _snap(page, frames, 2.5, fps)

        page.locator(".example-chip").nth(1).click()
        page.locator("#classify-button").click()
        page.wait_for_selector('#result[data-state="ready"]', timeout=15_000)
        _snap(page, frames, 3.0, fps)

        context.close()
        browser.close()

    payload = _encode(frames, width=width, fps=fps, colors=colors)
    for fallback_colors in (96, 64, 48):
        if len(payload) <= MAX_GIF_BYTES:
            break
        payload = _encode(frames, width=width, fps=fps, colors=fallback_colors)
    if len(payload) > MAX_GIF_BYTES:
        payload = _encode(frames, width=round(width * 0.8), fps=fps, colors=48)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    print(
        f"{out_path}: {len(payload) / 1_000_000:.2f} MB, {len(frames)} frames, "
        f"{fps} fps, width {width}"
    )
    if len(payload) > MAX_GIF_BYTES:
        raise SystemExit("the generated GIF is still above the 3 MB budget")
    return out_path


def main() -> int:
    """Parse arguments and build the demo GIF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Output GIF path.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Model bundle to serve (default: the committed tiny test bundle).",
    )
    parser.add_argument("--width", type=int, default=880, help="Output width in pixels.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second.")
    parser.add_argument("--colors", type=int, default=128, help="GIF palette size.")
    arguments = parser.parse_args()
    record(
        arguments.out.resolve(),
        arguments.bundle.resolve(),
        width=arguments.width,
        fps=arguments.fps,
        colors=arguments.colors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
