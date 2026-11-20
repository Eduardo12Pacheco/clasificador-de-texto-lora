#!/usr/bin/env python3
"""Check WCAG AA contrast for the UI design tokens declared in ``app.css``."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CONTRAST_PAIRS",
    "PairCheck",
    "check_stylesheet",
    "contrast_ratio",
    "parse_tokens",
    "relative_luminance",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSS = REPO_ROOT / "src" / "stc" / "adapters" / "web" / "static" / "app.css"

_TOKEN = re.compile(r"--([\w-]+):\s*(#[0-9a-fA-F]{3,8})\s*;")
_ROOT_BLOCK = re.compile(r":root\s*\{(?P<body>.*?)\}", re.DOTALL)
_DARK_BLOCK = re.compile(r'\[data-theme="dark"\]\s*\{(?P<body>.*?)\}', re.DOTALL)
_SYSTEM_DARK_BLOCK = re.compile(
    r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{\s*"
    r'\[data-theme="system"\]\s*\{(?P<body>.*?)\}',
    re.DOTALL,
)

# (label, foreground token, background token, minimum ratio). Text pairs use 4.5:1;
# status colours and the focus ring are UI components and use 3:1.
CONTRAST_PAIRS: tuple[tuple[str, str, str, float], ...] = (
    ("text on background", "text", "bg", 4.5),
    ("muted text on background", "text-muted", "bg", 4.5),
    ("accent contrast on accent", "accent-contrast", "accent", 4.5),
    ("warning on surface", "warning", "surface", 3.0),
    ("danger on surface", "danger", "surface", 3.0),
    ("success on surface", "success", "surface", 3.0),
    ("focus ring on background", "focus", "bg", 3.0),
)


@dataclass(frozen=True, slots=True)
class PairCheck:
    """One contrast assertion between two declared tokens."""

    theme: str
    label: str
    foreground: str
    background: str
    minimum: float
    ratio: float

    @property
    def passed(self) -> bool:
        """Return whether the measured ratio meets the threshold."""
        return self.ratio >= self.minimum


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    digits = value.lstrip("#")
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    if len(digits) == 8:
        digits = digits[:6]
    if len(digits) != 6:
        raise ValueError(f"unsupported colour literal: {value}")
    return tuple(int(digits[index : index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore[return-value]


def relative_luminance(colour: str) -> float:
    """Return the WCAG 2.x relative luminance of a hex colour."""

    def linearise(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearise(channel) for channel in _hex_to_rgb(colour))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio between two hex colours."""
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _block_tokens(block: re.Match[str] | None) -> dict[str, str]:
    if block is None:
        return {}
    return dict(_TOKEN.findall(block.group("body")))


def parse_tokens(css: str) -> dict[str, dict[str, str]]:
    """Return ``{"light": {...}, "dark": {...}}`` parsed from the CSS token blocks."""
    light = _block_tokens(_ROOT_BLOCK.search(css))
    dark = _block_tokens(_DARK_BLOCK.search(css))
    system_dark = _block_tokens(_SYSTEM_DARK_BLOCK.search(css))
    if not light or not dark:
        raise ValueError("app.css must declare :root and [data-theme=dark] token blocks")
    if system_dark and system_dark != dark:
        raise ValueError('[data-theme="system"] dark tokens must match the explicit dark theme')
    return {"light": light, "dark": dark}


def check_stylesheet(css: str) -> list[PairCheck]:
    """Return every contrast assertion for both themes."""
    tokens = parse_tokens(css)
    results: list[PairCheck] = []
    for theme in ("light", "dark"):
        palette = tokens[theme]
        for label, foreground_token, background_token, minimum in CONTRAST_PAIRS:
            try:
                foreground = palette[foreground_token]
                background = palette[background_token]
            except KeyError as error:
                raise ValueError(f"{theme} theme is missing token {error.args[0]}") from error
            results.append(
                PairCheck(
                    theme=theme,
                    label=label,
                    foreground=foreground,
                    background=background,
                    minimum=minimum,
                    ratio=contrast_ratio(foreground, background),
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    """Validate the stylesheet and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--css", type=Path, default=DEFAULT_CSS, help="Stylesheet to inspect.")
    arguments = parser.parse_args(argv)
    results = check_stylesheet(arguments.css.read_text(encoding="utf-8"))
    failures = [result for result in results if not result.passed]
    for result in results:
        status = "OK" if result.passed else "FAIL"
        print(
            f"{result.theme:5} {result.label:30} {result.ratio:5.2f}:1 "
            f">= {result.minimum:<4} {status}"
        )
    if failures:
        print(
            f"FAIL: {len(failures)} of {len(results)} contrast pairs are below threshold",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {len(results)} contrast pairs checked ({len(results) // 2} per theme).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
