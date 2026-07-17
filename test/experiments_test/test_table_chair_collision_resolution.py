from __future__ import annotations

from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    build_free_table_query,
    sample_chair_count,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def test_build_free_table_query_creates_the_requested_number_of_chair_slots() -> None:
    """
    build_free_table_query must create free EGChair slots and must not condition
    any chair's relative pose.
    """
    params = UnderspecifiedParameters(build_free_table_query(3))
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    assert not any("relative_pose" in name for name in conditioned_names)


# ---------------------------------------------------------------------------
# sample_chair_count
# ---------------------------------------------------------------------------


def test_sample_chair_count_only_returns_values_from_the_training_distribution() -> None:
    """
    sample_chair_count draws from the empirical distribution of chair counts
    observed in training data (marginalising the fitted circuit down to a
    single aggregation-statistic variable is not supported by the underlying
    JPT), so every draw must be one of the values actually observed in
    training.
    """
    training_counts = [2, 3, 4, 4, 6]
    results = {sample_chair_count(training_counts) for _ in range(50)}
    assert results <= set(training_counts)


def test_sample_chair_count_returns_the_only_value_when_training_is_uniform() -> None:
    """
    With a single observed chair count, sample_chair_count must always return
    that value.
    """
    assert sample_chair_count([4]) == 4
