"""The ES/EN string catalogues stay complete and identical in shape (S15)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
I18N_JS = REPO_ROOT / "src" / "stc" / "adapters" / "web" / "static" / "i18n.js"
INDEX_HTML = REPO_ROOT / "src" / "stc" / "adapters" / "web" / "static" / "index.html"

_I18N_ATTRIBUTE = re.compile(r'data-i18n(?:-aria-label|-title|-placeholder)?="([^"]+)"')
_EXPECTED_SCENARIOS = {
    "alarm",
    "audio",
    "calendar",
    "cooking",
    "datetime",
    "email",
    "general",
    "home",
    "iot",
    "lists",
    "music",
    "news",
    "play",
    "qa",
    "recommendation",
    "social",
    "takeaway",
    "transport",
}


def _load_strings() -> dict[str, Any]:
    text = I18N_JS.read_text(encoding="utf-8")
    marker = "export const STRINGS = "
    start = text.index(marker) + len(marker)
    value, _ = json.JSONDecoder().raw_decode(text[start:].lstrip())
    return value


def _keys(path: tuple[str, ...], value: Any) -> set[tuple[str, ...]]:
    if isinstance(value, dict):
        found: set[tuple[str, ...]] = set()
        for key, child in value.items():
            found |= _keys((*path, key), child)
        return found
    return {path}


def test_locales_have_identical_key_sets() -> None:
    strings = _load_strings()

    assert set(strings) == {"es", "en"}
    assert _keys((), strings["es"]) == _keys((), strings["en"])


def test_every_key_used_by_the_page_exists_in_both_locales() -> None:
    strings = _load_strings()
    used = set(_I18N_ATTRIBUTE.findall(INDEX_HTML.read_text(encoding="utf-8")))

    assert used
    for locale in ("es", "en"):
        catalogue = strings[locale]
        missing = sorted(key for key in used if not _has_key(catalogue, key))
        assert missing == [], f"{locale} is missing {missing}"


def test_scenario_labels_cover_the_eighteen_massive_scenarios() -> None:
    strings = _load_strings()

    for locale in ("es", "en"):
        scenarios = strings[locale]["scenarios"]
        assert set(scenarios) == _EXPECTED_SCENARIOS
        assert all(label.strip() for label in scenarios.values())


def _has_key(catalogue: dict[str, Any], dotted: str) -> bool:
    current: Any = catalogue
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return isinstance(current, str)
