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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(
    tmp_path: Path,
    *,
    readme: Path | None = None,
    readme_es: Path | None = None,
    registry: Path | None = None,
):
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
    if registry is not None:
        command += ["--registry", str(registry)]
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
    readme = tmp_path / "README.en.md"
    readme_es = tmp_path / "README.md"
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
    readme = tmp_path / "README.en.md"
    readme.write_text("# no markers\n", encoding="utf-8")

    result = _run(tmp_path, readme=readme)

    assert result.returncode == 1
    assert "marker" in result.stderr


def test_package_renderer_and_cli_render_share_the_same_output(
    tmp_path: Path, final_reports
) -> None:
    from typer.testing import CliRunner

    from stc.adapters.cli.app import app
    from stc.adapters.reporting.results import render_results

    final_reports()
    direct_summary = tmp_path / "direct" / "summary.json"
    direct_table = tmp_path / "direct" / "results_table.md"
    render_results(
        summary_path=direct_summary,
        table_path=direct_table,
        final_dir=tmp_path / "final",
    )

    cli_dir = tmp_path / "cli"
    cli_summary = cli_dir / "summary.json"
    cli_table = cli_dir / "results_table.md"
    readme = cli_dir / "README.en.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(PREFIX, encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "report",
            "render",
            "--summary",
            str(cli_summary),
            "--table",
            str(cli_table),
            "--readme",
            str(readme),
            "--final-dir",
            str(tmp_path / "final"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert cli_summary.read_text(encoding="utf-8") == direct_summary.read_text(encoding="utf-8")
    assert cli_table.read_text(encoding="utf-8") == direct_table.read_text(encoding="utf-8")
    assert "### Results" in readme.read_text(encoding="utf-8")


def test_runtime_and_bundle_details_are_in_summary_and_readme(
    tmp_path: Path, final_reports
) -> None:
    final_reports()
    _write_json(
        tmp_path / "final" / "benchmark.json",
        {
            "schema_version": 1,
            "cpu_model": "Test CPU",
            "onnxruntime_version": "1.32.0",
            "threads": [
                {
                    "threads": 4,
                    "p50_ms": 15.8014,
                    "p95_ms": 19.8951,
                    "batch32_throughput_per_s": 93.1966,
                },
            ],
        },
    )
    registry = tmp_path / "model_registry.json"
    _write_json(
        registry,
        {
            "registry_version": 1,
            "models": [
                {
                    "name": "stc-es-intent",
                    "version": "0.1.0",
                    "size_bytes": 570007756,
                    "url": "https://example.invalid/model.zip",
                    "sha256": "a" * 64,
                    "license": "Apache-2.0",
                }
            ],
        },
    )
    readme = tmp_path / "README.en.md"
    readme.write_text(PREFIX, encoding="utf-8")
    readme_es = tmp_path / "README.md"
    readme_es.write_text(PREFIX, encoding="utf-8")

    result = _run(tmp_path, readme=readme, readme_es=readme_es, registry=registry)

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["runtime"] == {
        "cpu_model": "Test CPU",
        "onnxruntime_version": "1.32.0",
        "threads": 4,
        "p50_ms": 15.8014,
        "p95_ms": 19.8951,
        "batch32_throughput_per_s": 93.1966,
    }
    assert summary["bundle"]["size_bytes"] == 570007756
    text = readme.read_text(encoding="utf-8")
    assert "15.8 ms p50 / 19.9 ms p95" in text
    assert "570.0 MB" in text
    assert "Test CPU" in text
    spanish = readme_es.read_text(encoding="utf-8")
    assert "Inferencia en CPU" in spanish
    assert "per utterance" not in spanish


def test_missing_runtime_and_registry_are_omitted(tmp_path: Path, final_reports) -> None:
    final_reports()

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "runtime" not in summary
    assert "bundle" not in summary
