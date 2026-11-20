#!/usr/bin/env python3
"""Capture the README CLI image from a real ``stc predict`` run (T15).

The SVG is produced by Rich's ``Console(record=True).save_svg`` with a
self-contained template: the default Rich template references a CDN font, and
the README must not trigger third-party requests.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rich.console import CONSOLE_SVG_FORMAT, Console
from typer.testing import CliRunner

import stc.adapters.cli.commands.predict as predict_module
from stc.adapters.cli.app import app

SELF_CONTAINED_FORMAT = re.sub(
    r"\n    @font-face \{.*?\n    \}\}",
    "",
    CONSOLE_SVG_FORMAT.replace(
        'font-family: "Fira Code";',
        "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;",
    ).replace("font-family: Fira Code, monospace;", "font-family: ui-monospace, monospace;"),
    flags=re.DOTALL,
)
PHRASE = "despiértame a las cinco de la mañana"


def capture(bundle: Path, out: Path) -> Path:
    """Render the real CLI table for one canonical phrase to an SVG file."""
    recorder = Console(record=True, width=110, force_terminal=True)
    predict_module.console = lambda _context: recorder  # type: ignore[assignment]
    result = CliRunner().invoke(
        app,
        ["predict", PHRASE, "--model", str(bundle)],
    )
    if result.exit_code != 0:
        raise SystemExit(f"stc predict failed with exit {result.exit_code}: {result.output}")
    out.parent.mkdir(parents=True, exist_ok=True)
    recorder.save_svg(str(out), title="stc predict", code_format=SELF_CONTAINED_FORMAT)
    rendered = out.read_text(encoding="utf-8")
    if re.search(r"""(?:url\(["']?https?://|href=["']https?://)""", rendered):
        raise SystemExit("the SVG must not reference external resources")
    print(f"{out}: {out.stat().st_size} bytes")
    return out


def main() -> int:
    """Parse arguments and write the CLI SVG."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True, help="Model bundle to query.")
    parser.add_argument("--out", type=Path, required=True, help="Output SVG path.")
    arguments = parser.parse_args()
    capture(arguments.bundle.resolve(), arguments.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
