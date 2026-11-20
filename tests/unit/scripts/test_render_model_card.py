"""The model-card renderer rejects placeholders its summary cannot fill."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "render_model_card.py"

SUMMARY = {
    "models": {
        "lora": {
            "test": {"accuracy": 0.87, "macro_f1": 0.86},
            "calibration": {"temperature": 1.5, "review_threshold": 0.5},
        }
    },
    "comparison": {"verdict": "transformer_better"},
}


def test_known_placeholders_are_filled(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(SUMMARY), encoding="utf-8")
    template = tmp_path / "template.md"
    template.write_text(
        "# Card\n\naccuracy {{models.lora.test.accuracy}}\n"
        "macro {{models.lora.test.macro_f1}}\n"
        "verdict {{comparison.verdict}}\n",
        encoding="utf-8",
    )
    out = tmp_path / "MODEL_CARD.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--template",
            str(template),
            "--summary",
            str(summary),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "accuracy 0.8700" in text
    assert "macro 0.8600" in text
    assert "verdict transformer_better" in text


def test_unknown_placeholder_fails_without_writing(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(SUMMARY), encoding="utf-8")
    template = tmp_path / "template.md"
    template.write_text("{{models.lora.missing}}\n", encoding="utf-8")
    out = tmp_path / "MODEL_CARD.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--template",
            str(template),
            "--summary",
            str(summary),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "unknown placeholder" in result.stderr
    assert not out.exists()


def test_repository_template_renders_from_generated_summary(tmp_path: Path, final_reports) -> None:
    final_reports()
    render_results = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_results.py"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--table",
            str(tmp_path / "results_table.md"),
            "--final-dir",
            str(tmp_path / "final"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert render_results.returncode == 0, render_results.stderr
    out = tmp_path / "MODEL_CARD.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--template",
            str(REPO_ROOT / "docs" / "model_card.template.md"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "{{" not in out.read_text(encoding="utf-8")
