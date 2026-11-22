"""README hero and image rules: static SVG heroes, alt text, badge cap and switcher (S40-S42)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HEROES = {
    "en": REPO_ROOT / "docs" / "assets" / "readme" / "hero.en.svg",
    "es": REPO_ROOT / "docs" / "assets" / "readme" / "hero.svg",
}
READMES = {"en": REPO_ROOT / "README.en.md", "es": REPO_ROOT / "README.md"}
OTHER = {"en": "README.md", "es": "README.en.md"}
SVG = "{http://www.w3.org/2000/svg}"
FORBIDDEN_SVG_TAGS = {
    f"{SVG}script",
    f"{SVG}foreignObject",
    f"{SVG}style",
    f"{SVG}animate",
    f"{SVG}animateTransform",
    f"{SVG}set",
}
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
ALT_RE = re.compile(r'\balt="([^"]*)"')
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(")
BADGE_RE = re.compile(r"\[!\[")


@pytest.mark.parametrize("language", ["en", "es"])
def test_hero_is_a_static_svg_with_a_1200_unit_canvas(language: str) -> None:
    root = ET.fromstring(HEROES[language].read_bytes())  # noqa: S314 - committed local asset

    assert root.tag == f"{SVG}svg"
    assert root.get("viewBox") == "0 0 1200 360"
    assert root.find(f"{SVG}title") is not None
    assert (root.findtext(f"{SVG}desc") or "").strip()
    elements = list(root.iter())
    assert not [el.tag for el in elements if el.tag in FORBIDDEN_SVG_TAGS]
    for element in elements:
        for name in ("href", "{http://www.w3.org/1999/xlink}href", "src"):
            value = element.get(name, "")
            assert not value.startswith(("http://", "https://")), value


@pytest.mark.parametrize("language", ["en", "es"])
def test_hero_text_is_never_smaller_than_a_supporting_label(language: str) -> None:
    root = ET.fromstring(HEROES[language].read_bytes())  # noqa: S314 - committed local asset

    sizes = [float(text.get("font-size", "0")) for text in root.iter(f"{SVG}text")]

    assert sizes
    assert min(sizes) >= 18


@pytest.mark.parametrize("language", ["en", "es"])
def test_every_readme_image_has_alt_text(language: str) -> None:
    text = READMES[language].read_text(encoding="utf-8")

    html_alts = [ALT_RE.search(tag) for tag in IMG_TAG_RE.findall(text)]
    markdown_alts = MARKDOWN_IMAGE_RE.findall(text)

    assert html_alts
    assert all(match and match.group(1).strip() for match in html_alts)
    assert markdown_alts
    assert all(alt.strip() for alt in markdown_alts)


@pytest.mark.parametrize("language", ["en", "es"])
def test_badges_stay_within_the_assessment_cap(language: str) -> None:
    text = READMES[language].read_text(encoding="utf-8")
    header = text.split("\n## ", maxsplit=1)[0]

    assert len(BADGE_RE.findall(header)) <= 4


@pytest.mark.parametrize("language", ["en", "es"])
def test_language_switcher_stays_on_the_third_line(language: str) -> None:
    lines = READMES[language].read_text(encoding="utf-8").splitlines()

    assert f"]({OTHER[language]})" in lines[2]


@pytest.mark.parametrize("language", ["en", "es"])
def test_readme_embeds_its_own_hero(language: str) -> None:
    text = READMES[language].read_text(encoding="utf-8")

    assert f"docs/assets/readme/{HEROES[language].name}" in text
