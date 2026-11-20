"""The ADR index lists every record and every record has the required sections."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "adr"
INDEX = ADR_DIR / "README.md"
REQUIRED_HEADINGS = ("Context", "Decision", "Alternatives", "Consequences")
INDEX_LINK = re.compile(r"\]\((\d{4}-[^)]+\.md)\)")
STATUS = re.compile(
    r"^- \*\*Status:\*\* (Proposed|Accepted|Superseded by ADR-\d{4})\s*$",
    re.MULTILINE,
)


def _adr_files() -> list[Path]:
    return sorted(path for path in ADR_DIR.glob("*.md") if path.name != "README.md")


def test_adr_numbers_0001_to_0009_exist() -> None:
    names = [path.name for path in _adr_files()]

    expected = {f"{number:04d}" for number in range(1, 10)}
    found = {name[:4] for name in names}
    assert found == expected


def test_index_links_every_adr_and_no_others() -> None:
    text = INDEX.read_text(encoding="utf-8")
    linked = set(INDEX_LINK.findall(text))
    files = {path.name for path in _adr_files()}

    assert linked == files
    for name in linked:
        assert (ADR_DIR / name).is_file()


def test_every_adr_has_status_and_required_sections() -> None:
    for path in _adr_files():
        text = path.read_text(encoding="utf-8")
        headings = {
            match.group(1).strip() for match in re.finditer(r"^## (.+)$", text, flags=re.MULTILINE)
        }

        assert STATUS.search(text), f"{path.name} needs a Status line"
        assert text.startswith("# ADR-"), f"{path.name} needs an ADR title"
        for heading in REQUIRED_HEADINGS:
            assert heading in headings, f"{path.name} needs a ## {heading} section"
