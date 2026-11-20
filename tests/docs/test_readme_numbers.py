"""README results blocks and quickstart stay in sync with committed reports (S20)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from stc.adapters.reporting.results import (
    END_MARKER,
    START_MARKER,
    render_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "reports" / "summary.json"
READMES = {"en": REPO_ROOT / "README.md", "es": REPO_ROOT / "README.es.md"}
QUICKSTART = [
    'uv tool install "spanish-text-classifier-lora @ '
    "https://github.com/Eduardo12Pacheco/spanish-text-classifier-lora/releases/download/"
    'v0.1.0/spanish_text_classifier_lora-0.1.0-py3-none-any.whl"',
    "stc model pull",
    'stc predict "despiértame a las cinco de la mañana"',
]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^)\s]+))(?:\s+[^)]*)?\)")


def _summary() -> dict[str, object]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _block(text: str, path: Path) -> str:
    assert text.count(START_MARKER) == 1, f"{path}: expected one {START_MARKER}"
    assert text.count(END_MARKER) == 1, f"{path}: expected one {END_MARKER}"
    inner = text.split(START_MARKER, maxsplit=1)[1].split(END_MARKER, maxsplit=1)[0]
    return inner.strip()


@pytest.mark.parametrize("language", ["en", "es"])
def test_results_block_matches_the_generated_render(language: str) -> None:
    path = READMES[language]
    text = path.read_text(encoding="utf-8")

    assert _block(text, path) == render_block(_summary(), language)


@pytest.mark.parametrize("language", ["en", "es"])
def test_readme_line_budget(language: str) -> None:
    lines = READMES[language].read_text(encoding="utf-8").splitlines()

    assert len(lines) <= 150


def test_readmes_have_the_same_sections() -> None:
    english = READMES["en"].read_text(encoding="utf-8").splitlines()
    spanish = READMES["es"].read_text(encoding="utf-8").splitlines()

    english_headings = [line for line in english if line.startswith("## ")]
    spanish_headings = [line for line in spanish if line.startswith("## ")]
    assert len(english_headings) == len(spanish_headings) >= 8


def test_quickstart_commands_appear_in_both_readmes() -> None:
    for language, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        for command in QUICKSTART:
            assert command in text, f"{path.name} ({language}) is missing {command!r}"


def test_relative_links_and_images_exist() -> None:
    missing: list[str] = []
    for path in READMES.values():
        for match in LINK_RE.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1) or match.group(2)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            if relative and not (path.parent / relative).exists():
                missing.append(f"{path.name}: {relative}")
    assert not missing, missing
