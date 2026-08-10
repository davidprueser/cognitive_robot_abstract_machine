from __future__ import annotations

import pytest

from experiments.scene_generation_experiments.utils import (
    MAXIMUM_LEAF_COUNT,
    MINIMUM_ROWS_PER_LEAF,
    min_samples_per_leaf_for,
)


def test_a_large_training_set_is_bounded_by_the_leaf_budget() -> None:
    """
    Grounding deep-copies a fitted circuit once per sampled part, so circuit
    size -- not statistical resolution -- is what has to be bounded once the
    training set is large.
    """
    fraction = min_samples_per_leaf_for(50_000)

    assert fraction == pytest.approx(1 / MAXIMUM_LEAF_COUNT)


def test_a_small_training_set_is_bounded_by_the_rows_per_leaf_floor() -> None:
    """
    Below the leaf budget the binding constraint flips: a leaf fitted on a
    handful of rows describes those rows rather than the distribution, so the
    fraction has to grow as the training set shrinks.
    """
    fraction = min_samples_per_leaf_for(200)

    assert fraction == pytest.approx(MINIMUM_ROWS_PER_LEAF / 200)
    assert fraction > 1 / MAXIMUM_LEAF_COUNT


def test_the_fraction_never_exceeds_the_whole_training_set() -> None:
    """
    A fraction above one would leave the circuit unable to split at all, which
    must not happen however few rows there are.
    """
    assert min_samples_per_leaf_for(1) <= 1.0
    assert min_samples_per_leaf_for(0) <= 1.0


@pytest.mark.parametrize("row_count", [10, 200, 5_000, 50_000, 500_000])
def test_the_fraction_is_always_usable(row_count: int) -> None:
    """
    Every training size must yield a fraction the circuit library accepts.
    """
    fraction = min_samples_per_leaf_for(row_count)

    assert 0.0 < fraction <= 1.0


def test_the_fraction_never_grows_with_the_training_set() -> None:
    """
    More data may buy finer leaves but never coarser ones.
    """
    fractions = [min_samples_per_leaf_for(n) for n in (100, 1_000, 10_000, 100_000)]

    assert fractions == sorted(fractions, reverse=True)
