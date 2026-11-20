"""The lexicon-coverage script refuses the test split and counts changed rows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "lexicon_coverage.py"


def _row(row_id: str, text: str) -> dict[str, str]:
    intent = "alarm_set"
    return {
        "id": row_id,
        "text": text,
        "intent": intent,
        "scenario": "alarm",
        "worker_id": "w1",
    }


def test_test_split_is_refused_with_exit_six() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--split", "test"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 6
    assert "test split is locked" in result.stderr
    assert "mx " not in result.stdout


def test_dev_coverage_counts_rows_changed_by_each_profile(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "massive-1.0"
    processed.mkdir(parents=True)
    rows = [
        _row("1", "Enciende el móvil ahora"),
        _row("2", "hola mundo"),
        _row("3", "Ponme música"),
    ]
    (processed / "dev.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--split", "dev", "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "split dev: 3 utterances" in result.stdout
    assert "mx 1" in result.stdout
    assert "ec 1" in result.stdout
    assert "ar 2" in result.stdout
