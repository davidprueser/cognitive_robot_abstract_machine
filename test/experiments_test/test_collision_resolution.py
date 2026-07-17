from __future__ import annotations

from experiments.scene_generation_experiments.collision_resolution import (
    minimal_resample_set,
)


# ---------------------------------------------------------------------------
# minimal_resample_set - greedy minimum vertex cover of the collision graph
# ---------------------------------------------------------------------------


def test_minimal_resample_set_picks_one_index_for_a_simple_pair() -> None:
    """
    For a single colliding pair, exactly one of the two indices must be chosen,
    deterministically (the higher one, by the tie-break rule).
    """
    assert minimal_resample_set({(0, 1)}) == {1}


def test_minimal_resample_set_picks_the_shared_index_for_a_star_collision() -> None:
    """
    When one index collides with two others that do not collide with each other,
    discarding just the shared index resolves every collision -- that minimal,
    single-index set must be returned, not a larger valid-but-wasteful cover.
    """
    assert minimal_resample_set({(0, 1), (0, 2)}) == {0}


def test_minimal_resample_set_is_empty_without_collisions() -> None:
    """
    With no colliding pairs, nothing needs resampling.
    """
    assert minimal_resample_set(set()) == set()
