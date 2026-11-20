"""Baseline grid parsing and the D5 winner-selection rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from stc.application.dto import BaselineCandidate, BaselineGrid
from stc.application.services.baseline_selection import select_baseline_winner
from stc.application.use_cases.train_baseline import load_baseline_grid

pytestmark = pytest.mark.unit


def candidate(
    *,
    macro_f1: float,
    accuracy: float,
    n_features: int = 100,
    label: str = "x",
) -> BaselineCandidate:
    return BaselineCandidate(
        params={"label": label},
        dev_accuracy=accuracy,
        dev_macro_f1=macro_f1,
        n_features=n_features,
        fit_seconds=0.1,
    )


def test_select_winner_breaks_ties_by_accuracy_then_features() -> None:
    low_accuracy = candidate(macro_f1=0.8, accuracy=0.80, n_features=10, label="a")
    high_accuracy = candidate(macro_f1=0.8, accuracy=0.90, n_features=20, label="b")
    fewer_features = candidate(macro_f1=0.8, accuracy=0.90, n_features=5, label="c")

    winner = select_baseline_winner([low_accuracy, high_accuracy, fewer_features])

    assert winner.params["label"] == "c"


def test_select_winner_prefers_highest_macro_f1() -> None:
    first = candidate(macro_f1=0.79, accuracy=0.95, n_features=1, label="a")
    second = candidate(macro_f1=0.81, accuracy=0.70, n_features=99, label="b")

    assert select_baseline_winner([first, second]).params["label"] == "b"


def test_select_winner_keeps_the_first_of_full_ties() -> None:
    first = candidate(macro_f1=0.8, accuracy=0.8, n_features=10, label="first")
    second = candidate(macro_f1=0.8, accuracy=0.8, n_features=10, label="second")

    assert select_baseline_winner([first, second]).params["label"] == "first"


def test_select_winner_rejects_an_empty_grid() -> None:
    from stc.domain.errors import InvalidInputError

    with pytest.raises(InvalidInputError):
        select_baseline_winner([])


def test_grid_default_matches_decision_d5() -> None:
    grid = BaselineGrid.default()

    assert len(grid.word_ngrams) == 2
    assert len(grid.char_ngrams) == 2
    assert len(grid.svc_c) == 5
    assert grid.class_weights == ("none", "balanced")
    assert len(grid.logreg_c) == 2
    # 2 word x 2 char x 5 C x 2 class weights = 40 SVC configs + 2 logistic.
    assert grid.n_svc_configs == 40
    assert grid.n_configs == 42


def test_grid_from_mapping_round_trips() -> None:
    grid = BaselineGrid.default()
    payload = {"grid": grid.to_dict()}

    assert BaselineGrid.from_mapping(payload["grid"]) == grid


@pytest.mark.parametrize(
    "mapping",
    [
        {"word_ngrams": []},
        {"svc_c": [0.0, 1.0]},
        {"class_weights": ["weird"]},
        {"char_ngrams": [[3, 2]]},
        {"random_state": "thirteen"},
    ],
)
def test_grid_rejects_invalid_values(mapping: dict[str, object]) -> None:
    from stc.domain.errors import InvalidInputError

    with pytest.raises(InvalidInputError):
        BaselineGrid.from_mapping(mapping)


def test_load_grid_reads_toml_and_uses_defaults_without_a_file(tmp_path: Path) -> None:
    config = tmp_path / "baseline.toml"
    config.write_text(
        "[grid]\nword_ngrams = [[1, 2]]\nchar_ngrams = [[2, 5]]\n"
        "svc_c = [0.5]\nclass_weights = ['none']\nlogreg_c = [10.0]\n"
        "random_state = 7\nquick_configs = 1\n",
        encoding="utf-8",
    )

    loaded = load_baseline_grid(config)

    assert loaded.random_state == 7
    assert loaded.n_configs == 2
    assert load_baseline_grid(None).n_configs == 42
