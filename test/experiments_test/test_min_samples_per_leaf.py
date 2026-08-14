from __future__ import annotations

import math

import pytest

from experiments.scene_generation_experiments.utils import (
    MAXIMUM_LEAF_COUNT,
    MINIMUM_ROWS_PER_LEAF,
    MINIMUM_ROWS_PER_LEAF_WHEN_DATA_IS_SPARSE,
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


@pytest.mark.parametrize("row_count", [5, 6, 11, 22, 49, 50])
def test_a_training_set_at_or_below_the_leaf_floor_still_yields_a_true_fraction(
    row_count: int,
) -> None:
    """
    A value of exactly ``1.0`` is read by
    :class:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree` as an
    *absolute* row count of one -- the least restrictive setting there is --
    rather than as "all of the training set", so a training set this small must
    never produce it: three real shelf types hold 5, 6, 11 and 22 rows apiece
    and each still needs genuine overfitting protection.
    """
    fraction = min_samples_per_leaf_for(row_count)

    assert fraction < 1.0


@pytest.mark.parametrize(
    "row_count,expected_absolute_rows",
    [(5, 4), (6, 5), (11, 5), (22, 5), (50, 5)],
)
def test_the_sparse_fallback_still_lets_uneven_sub_populations_form_their_own_leaf(
    row_count: int, expected_absolute_rows: int
) -> None:
    """
    The three real shelf types this was written for hold 5, 6 and 11 rows: each
    must still be able to earn a leaf of its own rather than being forced to
    share one with a differently sized type, however few rows any of them has.
    """
    fraction = min_samples_per_leaf_for(row_count)

    assert math.ceil(fraction * row_count) == expected_absolute_rows


def test_the_sparse_fallback_never_exceeds_its_own_target() -> None:
    """
    :data:`MINIMUM_ROWS_PER_LEAF_WHEN_DATA_IS_SPARSE` is the ceiling once the
    training set is smaller than :data:`MINIMUM_ROWS_PER_LEAF`; the resolved
    absolute row count must never demand more than that.
    """
    for row_count in range(2, MINIMUM_ROWS_PER_LEAF + 1):
        fraction = min_samples_per_leaf_for(row_count)

        assert math.ceil(fraction * row_count) <= MINIMUM_ROWS_PER_LEAF_WHEN_DATA_IS_SPARSE
