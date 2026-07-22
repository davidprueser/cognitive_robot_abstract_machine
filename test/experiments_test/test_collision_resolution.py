from __future__ import annotations

from experiments.scene_generation_experiments.collision_resolution import (
    _build_free_object2d_query,
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


# ---------------------------------------------------------------------------
# _build_free_object2d_query - free floor object sampling query
# ---------------------------------------------------------------------------


def test_build_free_object2d_query_pins_roll_and_pitch_to_upright() -> None:
    """
    Free floor objects always sit upright without tilting (only yaw varies),
    so roll and pitch must be fixed evidence rather than left underspecified.
    A degenerate (always-constant) circuit dimension left underspecified
    leaks the query's ``...`` placeholder straight through the sample instead
    of resolving it to a number, so only yaw -- which genuinely varies in the
    training data -- may be left for the RSPN to sample.
    """
    orientation = _build_free_object2d_query().kwargs["orientation"]

    assert orientation.kwargs["x"] == 0.0
    assert orientation.kwargs["y"] == 0.0
    assert orientation.kwargs["z"] is ...
