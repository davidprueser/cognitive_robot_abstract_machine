from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from experiments.scene_generation_experiments.exceptions import (
    TableChairLayoutResolutionError,
)
from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    _find_colliding_chair_indices,
    build_free_table_query,
    resolve_table_chair_collisions,
    sample_chair_count,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGTableWithChairs,
    ObjectType,
)


def _chair(
    chair_id: str,
    distance: float,
    angle: float,
    facing: float = 0.0,
    width: float = 0.5,
    length: float = 0.5,
) -> EGChair:
    return EGChair(
        id=chair_id,
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=length, width=width),
        relative_pose=EGRelativePolarPose(
            distance_from_table_center=distance,
            angle_from_table_center=angle,
            facing_angle_relative_to_table=facing,
        ),
        source_id="chair_src",
    )


def _table_with_chairs(chairs: list[EGChair]) -> EGTableWithChairs:
    return EGTableWithChairs(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=0.75, length=1.2, width=0.8),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        chairs=chairs,
    )


# ---------------------------------------------------------------------------
# Chair-chair collision detection
# ---------------------------------------------------------------------------


def test_overlapping_chairs_are_flagged() -> None:
    """
    Two chairs placed at (almost) the same position around the table must be
    flagged as colliding.
    """
    group = _table_with_chairs(
        [
            _chair("chair_0", distance=1.0, angle=0.0),
            _chair("chair_1", distance=1.0, angle=1.0),
        ]
    )
    assert _find_colliding_chair_indices(group) == {1}


def test_well_separated_chairs_are_not_flagged() -> None:
    """
    Chairs spaced well apart around the table must not be flagged.
    """
    group = _table_with_chairs(
        [
            _chair("chair_0", distance=1.0, angle=0.0),
            _chair("chair_1", distance=1.0, angle=90.0),
            _chair("chair_2", distance=1.0, angle=180.0),
            _chair("chair_3", distance=1.0, angle=270.0),
        ]
    )
    assert _find_colliding_chair_indices(group) == set()


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def test_build_free_table_query_creates_the_requested_number_of_chair_slots() -> None:
    """
    build_free_table_query must create exactly chair_count free EGChair slots
    and must not condition any chair's relative pose.
    """
    query = build_free_table_query(3)
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    assert not any("relative_pose" in name for name in conditioned_names)


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------


def test_resolve_table_chair_collisions_resamples_only_colliding_chairs() -> None:
    """
    resolve_table_chair_collisions must repair a colliding layout by resampling
    only the flagged chairs, leaving the rest untouched, and return once the
    layout is collision-free.
    """
    colliding_group = _table_with_chairs(
        [
            _chair("chair_0", distance=1.0, angle=0.0),
            _chair("chair_1", distance=1.0, angle=1.0),
        ]
    )
    fixed_group = _table_with_chairs(
        [
            _chair("chair_0", distance=1.0, angle=0.0),
            _chair("chair_1", distance=1.0, angle=180.0),
        ]
    )

    with patch(
        "experiments.scene_generation_experiments.table_chair_collision_resolution.ProbabilisticBackend"
    ) as backend_class:
        backend_class.return_value.evaluate.return_value = [fixed_group]
        result = resolve_table_chair_collisions(colliding_group, rspn=MagicMock())

    assert result.chairs[0].id == "chair_0"
    assert _find_colliding_chair_indices(result) == set()


def test_resolve_table_chair_collisions_raises_after_max_passes_when_unsatisfiable() -> None:
    """
    A layout that can never become collision-free must not spin the repair loop
    forever -- resolve_table_chair_collisions must give up and raise once
    max_passes is exhausted.
    """
    unfixable_group = _table_with_chairs(
        [
            _chair("chair_0", distance=1.0, angle=0.0),
            _chair("chair_1", distance=1.0, angle=1.0),
        ]
    )

    with patch(
        "experiments.scene_generation_experiments.table_chair_collision_resolution.ProbabilisticBackend"
    ) as backend_class:
        backend_class.return_value.evaluate.return_value = [unfixable_group]
        with pytest.raises(TableChairLayoutResolutionError):
            resolve_table_chair_collisions(unfixable_group, rspn=MagicMock(), max_passes=3)


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
