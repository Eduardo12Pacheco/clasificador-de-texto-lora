"""The results renderer builds the summary and rewrites only the marker block."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "render_results.py"

PREFIX = (
    "# Project\n\nIntro stays byte-identical.\n\n"
    "<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n\nAfter stays too.\n"
)


def _run(tmp_path: Path, *, readme: Path | None = None, readme_es: Path | None = None):
    command = [
        sys.executable,
        str(SCRIPT),
        "--summary",
        str(tmp_path / "summary.json"),
        "--table",
        str(tmp_path / "results_table.md"),
        "--final-dir",
        str(tmp_path / "final"),
    ]
    if readme is not None:
        command += ["--readme", str(readme)]
    if readme_es is not None:
        command += ["--readme-es", str(readme_es)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_summary_and_table_are_built_from_final_reports(tmp_path: Path, final_reports) -> None:
    final_reports()

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["baseline"]["test"]["macro_f1"] == 0.84
    assert summary["models"]["lora"]["test"]["macro_f1"] == 0.86
    assert summary["comparison"]["verdict"] == "transformer_better"
    assert summary["models"]["lora"]["calibration"]["ece_after"] == 0.04
    assert set(summary["models"]["lora"]["robustness"]) == {"uppercase", "regionalize:ar"}
    table = (tmp_path / "results_table.md").read_text(encoding="utf-8")
    assert "TF-IDF baseline" in table
    assert "LoRA transformer" in table
    assert "0.8600" in table


def test_readme_blocks_are_replaced_without_touching_the_outside(
    tmp_path: Path, final_reports
) -> None:
    final_reports()
    readme = tmp_path / "README.md"
    readme_es = tmp_path / "README.es.md"
    readme.write_text(PREFIX, encoding="utf-8")
    readme_es.write_text(PREFIX, encoding="utf-8")

    result = _run(tmp_path, readme=readme, readme_es=readme_es)

    assert result.returncode == 0, result.stderr
    english = readme.read_text(encoding="utf-8")
    spanish = readme_es.read_text(encoding="utf-8")
    assert english.startswith("# Project\n\nIntro stays byte-identical.\n\n")
    assert english.endswith("<!-- RESULTS:END -->\n\nAfter stays too.\n")
    assert "### Results" in english
    assert "### Resultados" in spanish
    assert "wins on this synthetic fixture" in english
    assert "gana en este fixture sintético" in spanish

    first = readme.read_bytes()
    second = _run(tmp_path, readme=readme, readme_es=readme_es)
    assert second.returncode == 0, second.stderr
    assert readme.read_bytes() == first


def test_outdated_summary_is_rebuilt(tmp_path: Path, final_reports) -> None:
    final_reports()
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    old = 1_000_000_000
    os.utime(tmp_path / "summary.json", (old, old))

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["lora"]["test"]["macro_f1"] == 0.86


def test_missing_markers_is_an_error(tmp_path: Path, final_reports) -> None:
    final_reports()
    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n", encoding="utf-8")

    result = _run(tmp_path, readme=readme)

    assert result.returncode == 1
    assert "marker" in result.stderr
