"""The experiment collector writes deterministic compact reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "collect_experiment_reports.py"
VALIDATE = REPO_ROOT / "scripts" / "validate_reports.py"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _artifacts(root: Path) -> None:
    _write(
        root / "environment.json",
        {
            "schema_version": 1,
            "git_sha": "a" * 40,
            "uv_lock_sha256": "b" * 64,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 4060",
            "driver_version": "620.70.06",
            "library_versions": {
                "torch": "2.15.0",
                "transformers": "4.57.6",
                "peft": "0.22.0",
                "optuna": "5.1.0",
                "mlflow": "3.18.0",
            },
        },
    )
    _write(
        root / "screening" / "screening.json",
        {
            "schema_version": 1,
            "runs": [
                {
                    "backbone": "BSC-LT/MrBERT-es",
                    "seed": 13,
                    "dev_macro_f1": 0.84,
                    "dev_accuracy": 0.85,
                    "trainable_params": 100,
                    "epoch_seconds": 20.0,
                    "peak_vram_mib": 1500.0,
                    "cpu_latency_ms_p50": 15.5,
                    "exportable": True,
                    "run_dir": "/home/owner/artifacts/screening/a",
                }
            ],
        },
    )
    _write(
        root / "screening" / "selection.json",
        {
            "schema_version": 1,
            "selected": "BSC-LT/MrBERT-es",
            "rule_branch": "mrbert_ahead",
            "per_backbone_means": {"BSC-LT/MrBERT-es": 0.84},
        },
    )
    _write(
        root / "full_ft" / "full_ft.json",
        {
            "schema_version": 1,
            "backbone": "BSC-LT/MrBERT-es",
            "runs": [{"lr": 3e-5, "dev_macro_f1": 0.85, "run": "x"}],
            "best": {"lr": 3e-5, "dev_macro_f1": 0.85},
        },
    )
    _write(
        root / "search" / "search_trials.json",
        {
            "schema_version": 1,
            "study_name": "stc-lora-search",
            "trials": [{"number": 0, "state": "COMPLETE", "value": 0.83, "params": {"r": 16}}],
            "best": {"value": 0.83, "trial": 0, "params": {"r": 16}},
        },
    )
    (root / "search" / "search_summary.md").write_text("# Search\n", encoding="utf-8")
    (root / "search" / "search_history.png").write_bytes(b"png")
    _write(
        root / "final" / "final_seeds.json",
        {
            "schema_version": 1,
            "seeds": [{"seed": 13, "dev_macro_f1": 0.84, "run": "x"}],
            "selected_seed": 13,
            "dev_macro_f1_mean": 0.84,
            "dev_macro_f1_std": 0.0,
        },
    )
    _write(
        root / "screening" / "bsc-lt-mrbert-es-seed-13" / "run.json",
        {"wall_seconds": 12.5},
    )


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_collector_writes_compact_reports_and_validates(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    out = tmp_path / "experiments"
    figures = tmp_path / "figures"
    _artifacts(artifacts)

    result = _run(
        SCRIPT,
        "--artifacts",
        str(artifacts),
        "--out",
        str(out),
        "--figures",
        str(figures),
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "environment.json",
        "screening.json",
        "selection.json",
        "full_ft_reference.json",
        "search_trials.json",
        "search_summary.md",
        "final_seeds.json",
    ):
        assert (out / name).is_file(), name
    assert (figures / "search_history.png").is_file()
    screening = json.loads((out / "screening.json").read_text(encoding="utf-8"))
    assert "run_dir" not in screening["runs"][0]
    environment = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert environment["stages_seconds"]["screening"] == 12.5

    validation = _run(VALIDATE, "experiments", "--root", str(out))
    assert validation.returncode == 0, validation.stderr + validation.stdout


def test_collector_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    out = tmp_path / "experiments"
    _artifacts(artifacts)
    _run(SCRIPT, "--artifacts", str(artifacts), "--out", str(out), "--figures", str(tmp_path / "f"))
    before = {path.name: path.read_bytes() for path in out.glob("*.json")}

    _run(SCRIPT, "--artifacts", str(artifacts), "--out", str(out), "--figures", str(tmp_path / "f"))

    assert {path.name: path.read_bytes() for path in out.glob("*.json")} == before


def test_collector_reports_missing_artifacts(tmp_path: Path) -> None:
    result = _run(
        SCRIPT,
        "--artifacts",
        str(tmp_path / "empty"),
        "--out",
        str(tmp_path / "out"),
        "--figures",
        str(tmp_path / "figures"),
    )

    assert result.returncode == 1
    assert "no experiment artifacts" in result.stderr
