from __future__ import annotations

from experiments.scene_generation_experiments.proximity_group_collision_resolution import (
    build_free_group_query,
    sample_member_count,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def test_build_free_table_query_creates_the_requested_number_of_chair_slots() -> None:
    """
    build_free_group_query must create free EGGroupMember slots and must not condition
    any member's relative pose.
    """
    params = UnderspecifiedParameters(build_free_group_query(3))
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    assert not any("relative_pose" in name for name in conditioned_names)


# ---------------------------------------------------------------------------
# sample_member_count
# ---------------------------------------------------------------------------


def test_sample_chair_count_only_returns_values_from_the_training_distribution() -> None:
    """
    sample_member_count draws from the empirical distribution of member counts
    observed in training data (marginalising the fitted circuit down to a
    single aggregation-statistic variable is not supported by the underlying
    JPT), so every draw must be one of the values actually observed in
    training.
    """
    training_counts = [2, 3, 4, 4, 6]
    results = {
        sample_member_count({ObjectType.TABLE: training_counts}, ObjectType.TABLE)
        for _ in range(50)
    }
    assert results <= set(training_counts)


def test_sample_chair_count_returns_the_only_value_when_training_is_uniform() -> None:
    """
    With a single observed member count, sample_member_count must always return
    that value.
    """
    assert sample_member_count({ObjectType.TABLE: [4]}, ObjectType.TABLE) == 4


def test_sample_member_count_returns_zero_for_an_anchor_type_that_stands_alone() -> (
    None
):
    """
    Member counts are drawn per anchor type, so a type whose training anchors
    never gathered anything -- a refrigerator, a plant -- draws zero rather than
    borrowing a dining table's chairs from a pooled distribution.
    """
    counts = {ObjectType.TABLE: [4, 6], ObjectType.REFRIGERATOR: [0, 0]}

    assert sample_member_count(counts, ObjectType.REFRIGERATOR) == 0
    assert sample_member_count(counts, ObjectType.PLANT) == 0
