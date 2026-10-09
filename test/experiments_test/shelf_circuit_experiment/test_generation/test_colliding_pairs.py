from __future__ import annotations

from experiments.shelf_generation_experiments.generation.colliding_pairs import (
    CollidingPairs,
)


# %% minimal resample set
def test_nothing_is_resampled_without_collisions() -> None:
    assert CollidingPairs().minimal_resample_set() == set()


def test_one_member_of_a_single_pair_is_resampled() -> None:
    pairs = CollidingPairs()
    pairs.add(0, 1)

    assert pairs.minimal_resample_set() == {1}


def test_the_member_shared_by_every_pair_is_resampled_alone() -> None:
    pairs = CollidingPairs()
    for other in (1, 2, 3):
        pairs.add(0, other)

    assert pairs.minimal_resample_set() == {0}


def test_the_order_pairs_are_reported_in_does_not_matter() -> None:
    forward = CollidingPairs()
    backward = CollidingPairs()
    for first, second in [(0, 1), (1, 2), (2, 3)]:
        forward.add(first, second)
        backward.add(second, first)

    assert forward.minimal_resample_set() == backward.minimal_resample_set()
