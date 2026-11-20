"""The error-analysis check ties documented ids to the audited error list."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_error_analysis.py"

CATEGORIES = ["Near-synonymous intents", "Open domain", "Slots only", "Label noise", "Truncated"]


def _errors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"errors": [{"example_id": str(index)} for index in range(1, 20)]}),
        encoding="utf-8",
    )


def _document(path: Path, ids: list[str]) -> None:
    sections = [f"### {title}\n\n" + ", ".join(f"`ex-{i}`" for i in ids) for title in CATEGORIES]
    path.write_text("# Error analysis\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")


def _run(doc: Path, errors: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--doc", str(doc), "--errors", str(errors)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_document_exits_zero(tmp_path: Path) -> None:
    errors = tmp_path / "errors.json"
    doc = tmp_path / "ERROR_ANALYSIS.md"
    _errors(errors)
    _document(doc, ["1", "2", "3"])

    result = _run(doc, errors)

    assert result.returncode == 0, result.stderr
    assert "OK error analysis" in result.stdout


def test_unknown_id_is_reported(tmp_path: Path) -> None:
    errors = tmp_path / "errors.json"
    doc = tmp_path / "ERROR_ANALYSIS.md"
    _errors(errors)
    _document(doc, ["1", "2", "999"])

    result = _run(doc, errors)

    assert result.returncode == 1
    assert "unknown example id ex-999" in result.stderr


def test_too_few_categories_are_reported(tmp_path: Path) -> None:
    errors = tmp_path / "errors.json"
    doc = tmp_path / "ERROR_ANALYSIS.md"
    _errors(errors)
    doc.write_text("### One\n\n`ex-1`, `ex-2`, `ex-3`\n", encoding="utf-8")

    result = _run(doc, errors)

    assert result.returncode == 1
    assert "at least 5" in result.stderr
