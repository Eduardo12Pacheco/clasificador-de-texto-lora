"""The D7 sqrt-inverse-frequency weights are pure and clipped."""

from __future__ import annotations

import numpy as np
import pytest

from stc.adapters.training.class_weighting import sqrt_inv_freq_weights
from stc.domain.errors import InvalidInputError


def test_weights_follow_the_median_ratio_and_clip() -> None:
    weights = sqrt_inv_freq_weights([1, 4, 16])

    assert weights.tolist() == pytest.approx([2.0, 1.0, 0.5])


def test_extreme_counts_are_clipped() -> None:
    weights = sqrt_inv_freq_weights([1, 1_000_000])

    assert weights[0] == pytest.approx(4.0)
    assert weights[1] == pytest.approx((500000.5 / 1_000_000) ** 0.5)


def test_mapping_form_and_empty_class() -> None:
    weights = sqrt_inv_freq_weights({0: 4, 1: 16, 3: 0})

    assert weights.shape == (4,)
    assert weights[0] == pytest.approx((10.0 / 4.0) ** 0.5)
    assert weights[3] == pytest.approx(4.0)
    assert weights[1] == pytest.approx((10.0 / 16.0) ** 0.5)


def test_empty_or_all_zero_counts_are_rejected() -> None:
    with pytest.raises(InvalidInputError):
        sqrt_inv_freq_weights([])
    with pytest.raises(InvalidInputError):
        sqrt_inv_freq_weights([0, 0])


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(InvalidInputError):
        sqrt_inv_freq_weights([-1, 2])


def test_result_is_a_float_array() -> None:
    assert np.asarray(sqrt_inv_freq_weights([2, 2])).dtype == np.float64
