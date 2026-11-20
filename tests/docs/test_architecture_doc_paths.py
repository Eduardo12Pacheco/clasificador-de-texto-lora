"""Every repository path quoted in the architecture guide must exist."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("src/", "tests/", "scripts/", "docs/", "configs/", "reports/", ".github/")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
PLACEHOLDER = re.compile(r"[*<>{}|]")


def _quoted_paths(text: str) -> set[str]:
    candidates: set[str] = set()
    for span in CODE_SPAN.findall(text):
        value = span.strip()
        if not value.startswith(DOMAINS):
            continue
        if PLACEHOLDER.search(value) or " " in value:
            continue
        candidates.add(value.rstrip("/"))
    return candidates


def test_all_quoted_paths_exist() -> None:
    text = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    quoted = _quoted_paths(text)

    assert len(quoted) >= 20, "the architecture guide should reference the real tree"
    missing = sorted(path for path in quoted if not (REPO_ROOT / path).exists())
    assert missing == []


def test_architecture_guide_has_the_expected_sections() -> None:
    text = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)

    assert len(headings) >= 8
    for required in (
        "Purpose and constraints",
        "Layers and dependency direction",
        "Module map",
        "Ports and adapters",
        "Patterns in use",
        "Data and inference lifecycle",
        "Test pyramid",
        "How the boundaries are enforced",
        "Extension recipes",
    ):
        assert required in headings
