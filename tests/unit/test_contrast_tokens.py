"""The declared design tokens meet WCAG AA contrast in both themes (S15)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_contrast.py"
CSS = REPO_ROOT / "src" / "stc" / "adapters" / "web" / "static" / "app.css"


def _check_contrast_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_contrast", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_declared_pair_passes_its_threshold() -> None:
    module = _check_contrast_module()

    results = module.check_stylesheet(CSS.read_text(encoding="utf-8"))

    assert len(results) >= 14
    failures = [result for result in results if not result.passed]
    assert failures == [], "\n".join(
        f"{result.theme} {result.label}: {result.ratio:.2f} < {result.minimum}"
        for result in failures
    )
    assert {result.theme for result in results} == {"light", "dark"}


def test_both_themes_declare_every_checked_token() -> None:
    module = _check_contrast_module()

    tokens = module.parse_tokens(CSS.read_text(encoding="utf-8"))

    required = {
        "bg",
        "surface",
        "surface-2",
        "text",
        "text-muted",
        "border",
        "accent",
        "accent-contrast",
        "success",
        "warning",
        "danger",
        "focus",
    }
    for theme in ("light", "dark"):
        assert required <= set(tokens[theme])
        assert all(value.startswith("#") for value in tokens[theme].values())
