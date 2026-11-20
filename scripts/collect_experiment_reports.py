#!/usr/bin/env python
"""Collect compact, committed experiment reports from large local artifacts.

The GPU artifacts under ``artifacts/`` stay git-ignored; this script keeps the
small summaries that T11 commits under ``reports/experiments`` and copies the
search figures into ``reports/figures``. Output is deterministic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "screening": "screening",
    "full_ft": "full_ft",
    "search": "search",
    "final": "final",
}


def _load(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"error: {path} must contain a JSON object")
    return payload


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strip_run_paths(payload: Mapping[str, object], key: str) -> dict[str, object]:
    """Drop absolute artifact paths while keeping the compact evidence."""
    cleaned = dict(payload)
    rows = cleaned.get(key)
    if isinstance(rows, list):
        cleaned[key] = [
            {name: value for name, value in row.items() if name != "run_dir"}
            if isinstance(row, dict)
            else row
            for row in rows
        ]
    return cleaned


def _wall_seconds(root: Path) -> float:
    total = 0.0
    for run_json in sorted(root.rglob("run.json")):
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("wall_seconds"), int | float):
            total += float(payload["wall_seconds"])
    return round(total, 3)


def collect(artifacts: Path, out: Path, figures: Path) -> list[Path]:
    """Write every compact report that exists in ``artifacts``."""
    written: list[Path] = []
    environment = _load(artifacts / "environment.json")
    if environment is not None:
        environment = dict(environment)
        environment["stages_seconds"] = {
            stage: _wall_seconds(artifacts / directory)
            for stage, directory in STAGES.items()
            if (artifacts / directory).is_dir()
        }
        _write(out / "environment.json", environment)
        written.append(out / "environment.json")

    screening = _load(artifacts / "screening" / "screening.json")
    if screening is not None:
        _write(out / "screening.json", _strip_run_paths(screening, "runs"))
        written.append(out / "screening.json")
    selection = _load(artifacts / "screening" / "selection.json")
    if selection is not None:
        _write(out / "selection.json", selection)
        written.append(out / "selection.json")

    full_ft = _load(artifacts / "full_ft" / "full_ft.json")
    if full_ft is not None:
        _write(out / "full_ft_reference.json", _strip_run_paths(full_ft, "runs"))
        written.append(out / "full_ft_reference.json")

    search = _load(artifacts / "search" / "search_trials.json")
    if search is not None:
        compact = dict(search)
        trials = compact.get("trials")
        if isinstance(trials, list):
            compact["trials"] = [
                {
                    name: value
                    for name, value in trial.items()
                    if name in {"number", "state", "value", "params", "duration_seconds"}
                }
                if isinstance(trial, dict)
                else trial
                for trial in trials
            ]
        _write(out / "search_trials.json", compact)
        written.append(out / "search_trials.json")
    summary = artifacts / "search" / "search_summary.md"
    if summary.is_file():
        (out / "search_summary.md").parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary, out / "search_summary.md")
        written.append(out / "search_summary.md")

    final_seeds = _load(artifacts / "final" / "final_seeds.json")
    if final_seeds is not None:
        _write(out / "final_seeds.json", _strip_run_paths(final_seeds, "seeds"))
        written.append(out / "final_seeds.json")

    figures.mkdir(parents=True, exist_ok=True)
    for name in ("search_history.png", "search_importance.png"):
        source = artifacts / "search" / name
        if source.is_file():
            destination = figures / name
            shutil.copyfile(source, destination)
            written.append(destination)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Collect the compact reports and print what was written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args(argv)
    written = collect(args.artifacts, args.out, args.figures)
    for path in written:
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    if not written:
        print("error: no experiment artifacts found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
