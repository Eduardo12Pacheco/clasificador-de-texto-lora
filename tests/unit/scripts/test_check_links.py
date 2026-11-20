"""The Markdown link checker rejects broken relative links and anchors."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_links.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_links", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_relative_links_and_anchors_pass(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        "# Top\n\n## Install steps\n\nSee [usage](usage.md#run-it) and [top](#top).\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "usage.md").write_text("## Run it\n\nBack to [guide](guide.md).\n")
    (tmp_path / "README.md").write_text("[Guide](docs/guide.md#install-steps)\n", encoding="utf-8")

    assert _module().check_tree(tmp_path) == []


def test_missing_target_is_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[gone](missing.md)\n", encoding="utf-8")

    errors = _module().check_tree(tmp_path)

    assert len(errors) == 1
    assert "missing target: missing.md" in errors[0]


def test_missing_anchor_is_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[bad](guide.md#nope)\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("## Present\n", encoding="utf-8")

    errors = _module().check_tree(tmp_path)

    assert len(errors) == 1
    assert "missing anchor #nope" in errors[0]


def test_external_links_and_code_blocks_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "[site](https://example.com/missing)\n"
        "[mail](mailto:someone@example.com)\n"
        "```markdown\n[ignored](missing.md)\n```\n",
        encoding="utf-8",
    )

    assert _module().check_tree(tmp_path) == []


def test_absolute_and_escaping_paths_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "[abs](/etc/passwd)\n[escape](../outside.md)\n",
        encoding="utf-8",
    )

    errors = _module().check_tree(tmp_path)

    assert any("absolute path" in error for error in errors)
    assert any("escapes the repository" in error for error in errors)


def test_duplicate_headings_get_github_style_suffixes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "## Notes\n\n## Notes\n\n[first](#notes) [second](#notes-1)\n",
        encoding="utf-8",
    )

    assert _module().check_tree(tmp_path) == []


def test_main_exit_codes(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "README.md").write_text("[gone](missing.md)\n", encoding="utf-8")

    assert module.main(["--root", str(tmp_path)]) == 1
    (tmp_path / "README.md").write_text("[self](#top)\n# Top\n", encoding="utf-8")
    assert module.main(["--root", str(tmp_path)]) == 0
