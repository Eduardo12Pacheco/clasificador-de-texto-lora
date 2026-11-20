"""The tracker factory parses specs and never imports adapters itself."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from stc.adapters.tracking.composite_tracker import CompositeTracker
from stc.adapters.tracking.null_tracker import NullTracker
from stc.application.services.tracker_factory import TrackerFactory, parse_tracker_names
from stc.domain.errors import InvalidInputError


class _Fake:
    def __init__(self, name: str) -> None:
        self.name = name

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        pass

    def log_params(self, params: Mapping[str, object]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def end_run(self) -> None:
        pass


def _factory() -> TrackerFactory:
    def builder(name: str):
        return lambda run_dir: _Fake(name)

    return TrackerFactory(
        {"a": builder("a"), "b": builder("b"), "null": lambda run_dir: NullTracker()},
        composite_builder=lambda children: CompositeTracker(children),
    )


def test_parse_names_handles_commas_and_empty_specs() -> None:
    assert parse_tracker_names("jsonl, mlflow") == ("jsonl", "mlflow")
    assert parse_tracker_names(["jsonl", " "]) == ("jsonl",)
    assert parse_tracker_names("") == ("null",)


def test_single_tracker_is_returned_directly(tmp_path: Path) -> None:
    tracker = _factory().create("a", tmp_path)

    assert isinstance(tracker, _Fake)
    assert tracker.name == "a"


def test_multiple_names_produce_a_composite(tmp_path: Path) -> None:
    tracker = _factory().create("a,b", tmp_path)

    assert isinstance(tracker, CompositeTracker)
    assert [child.name for child in tracker.children] == ["a", "b"]  # type: ignore[attr-defined]


def test_unknown_name_is_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="unknown tracker"):
        _factory().create("a,nope", tmp_path)


def test_empty_spec_uses_the_null_object(tmp_path: Path) -> None:
    assert isinstance(_factory().create("", tmp_path), NullTracker)


def test_names_sequence_is_accepted(tmp_path: Path) -> None:
    builders: dict[str, object] = {"a": lambda run_dir: _Fake("a")}
    factory = TrackerFactory(builders, composite_builder=lambda children: children[0])  # type: ignore[arg-type,dict-item]
    tracker = factory.create(["a"], tmp_path)

    assert isinstance(tracker, _Fake)
