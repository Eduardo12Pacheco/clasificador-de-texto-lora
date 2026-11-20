"""Tracker adapters record events and isolate composite failures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from stc.adapters.tracking.composite_tracker import CompositeTracker
from stc.adapters.tracking.jsonl_tracker import JsonlTracker
from stc.adapters.tracking.null_tracker import NullTracker


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        self.calls.append("start")

    def log_params(self, params: Mapping[str, object]) -> None:
        self.calls.append("params")

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.calls.append(f"metrics:{step}")

    def log_artifact(self, path: Path) -> None:
        self.calls.append("artifact")

    def end_run(self) -> None:
        self.calls.append("end")


class _Failing:
    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        raise RuntimeError("boom")

    def log_params(self, params: Mapping[str, object]) -> None:
        raise RuntimeError("boom")

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        raise RuntimeError("boom")

    def log_artifact(self, path: Path) -> None:
        raise RuntimeError("boom")

    def end_run(self) -> None:
        raise RuntimeError("boom")


def test_jsonl_tracker_writes_one_event_per_line(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "tracker.jsonl"
    tracker = JsonlTracker(path)

    tracker.start_run("seed-13", {"backbone": "tiny"})
    tracker.log_params({"lr": 0.001})
    tracker.log_metrics({"dev_macro_f1": 0.5}, step=1)
    tracker.log_artifact(tmp_path / "run.json")
    tracker.end_run()

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "start",
        "params",
        "metrics",
        "artifact",
        "end",
    ]
    assert events[0]["tags"] == {"backbone": "tiny"}
    assert events[2]["step"] == 1


def test_null_tracker_accepts_everything(tmp_path: Path) -> None:
    tracker = NullTracker()
    tracker.start_run("run", {})
    tracker.log_params({})
    tracker.log_metrics({}, step=None)
    tracker.log_artifact(tmp_path)
    tracker.end_run()


def test_composite_fans_out_and_isolates_errors(tmp_path: Path) -> None:
    good = _Recorder()
    composite = CompositeTracker([_Failing(), good])

    composite.start_run("run", {})
    composite.log_params({"a": 1})
    composite.log_metrics({"m": 1.0}, step=2)
    composite.log_artifact(tmp_path)
    composite.end_run()

    assert good.calls == ["start", "params", "metrics:2", "artifact", "end"]
    assert len(composite.errors) == 5
    assert composite.children == (_Failing(), good) or len(composite.children) == 2


@pytest.mark.torch
def test_mlflow_tracker_logs_to_a_sqlite_store(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from stc.adapters.tracking.mlflow_tracker import MlflowTracker

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MlflowTracker(uri, experiment_name="stc-test")
    tracker.start_run("seed-13", {"backbone": "tiny"})
    tracker.log_params({"lr": 0.001})
    tracker.log_metrics({"dev_macro_f1": 0.5}, step=1)
    tracker.end_run()

    runs = mlflow.search_runs(experiment_names=["stc-test"], output_format="list")
    assert len(runs) == 1
    assert runs[0].data.tags["backbone"] == "tiny"
