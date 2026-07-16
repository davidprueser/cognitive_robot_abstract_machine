from __future__ import annotations

import random

from experiments.scene_generation_experiments.collision_resolution import (
    _find_colliding_indices,
)
from experiments.scene_generation_experiments.exceptions import (
    TableChairLayoutResolutionError,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGObject2D,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGShelfLayer,
    EGScale,
    EGTableWithChairs,
)


def sample_chair_count(training_chair_counts: list[int]) -> int:
    """
    Draw the number of chairs to sample around a table from the empirical
    distribution of chair counts observed in the training data.

    Chair count has to be drawn before the sampling query is built,
    since an exchangeable relation's list length is a structural
    property of the query, not something ``ground()`` can pick on its
    own. Drawing from the empirical distribution and then conditioning
    the query on that count (via the ``total_count`` aggregation
    statistic) is equivalent to ``P(count) * P(scale, poses | count)``
    -- the same joint the RSPN learned, just factored so that sampling
    ``count`` does not depend on marginalising the fitted circuit down
    to a single aggregation variable, which the underlying JPT does not
    support for aggregation statistics.

    :param training_chair_counts: Number of chairs observed per training
        table, e.g. ``[len(group.chairs) for group in
        table_chair_groups]``.
    :return: A chair count of at least one, drawn from the training
        distribution.
    """
    return random.choice(training_chair_counts)


def _project_chair_to_object2d(chair: EGChair) -> EGObject2D:
    """
    Project a chair's table-relative polar pose into a Cartesian
    :class:`EGObject2D` in the table's own local frame, so chair-chair
    collisions can be checked with the same box-proxy machinery already used
    for shelf objects.

    :param chair: The chair to project.
    :return: A Cartesian stand-in for *chair*, positioned in the table's
        local frame.
    """
    local_x, local_y, chair_yaw_relative_to_table = (
        chair.relative_pose.to_absolute_pose(0.0, 0.0, 0.0)
    )
    return EGObject2D(
        id=chair.id,
        room_id=chair.room_id,
        place_id="floor",
        object_type=chair.object_type,
        scale=chair.scale,
        position=EGPoint2D(x=local_x, y=local_y),
        orientation=EGRotation(x=0.0, y=0.0, z=chair_yaw_relative_to_table),
        source_id=chair.source_id,
    )


def _find_colliding_chair_indices(table_with_chairs: EGTableWithChairs) -> set[int]:
    """
    Return indices of chairs that collide with another chair around the table.

    Chairs are never checked against the table's own footprint: measured
    against real training data, chairs assigned to a table never overlap
    its bounding box, so treating the table as a collision obstacle
    would only risk rejecting plausible layouts the fitted distribution
    already produces.

    :param table_with_chairs: The table-with-chairs group to inspect.
    :return: Set of indices (into ``table_with_chairs.chairs``) that
        must be replaced.
    """
    proxy_layer = EGShelfLayer(
        scale=table_with_chairs.scale,
        objects=[
            _project_chair_to_object2d(chair) for chair in table_with_chairs.chairs
        ],
    )
    return _find_colliding_indices(proxy_layer)


def _build_free_chair_query():
    """
    Build a fully underspecified EGChair query with all pose fields free.

    :return: An underspecified EGChair with scale and relative_pose
        unset.
    """
    return underspecified(EGChair)(
        id=None,
        room_id=None,
        object_type=...,
        scale=underspecified(EGScale)(width=..., length=..., height=...),
        relative_pose=underspecified(EGRelativePolarPose)(
            distance_from_table_center=...,
            angle_from_table_center=...,
            facing_angle_relative_to_table=...,
        ),
        source_id=None,
    )


def _build_conditioned_table_query(
    fixed_chairs: list[EGChair],
    free_count: int,
    table_position: EGPoint2D | None = None,
    table_scale: EGScale | None = None,
    table_orientation: EGRotation | None = None,
):
    """
    Build an EGTableWithChairs query conditioning on fixed_chairs' poses and
    leaving free_count chair slots fully underspecified.

    :param fixed_chairs: Concrete EGChair instances whose relative pose
        is fixed as conditioning evidence.
    :param free_count: Number of fully-underspecified chair slots to
        resample.
    :param table_position: When provided, fixes the table's own position
        as conditioning evidence, so a repair pass samples chairs for
        the same table instance rather than an entirely new one.
    :param table_scale: When provided, fixes the table's own scale as
        conditioning evidence.
    :param table_orientation: When provided, fixes the table's own
        orientation as conditioning evidence.
    :return: An underspecified EGTableWithChairs query ready for
        ProbabilisticBackend evaluation.
    """

    def _fixed_slot(chair: EGChair):
        return underspecified(EGChair)(
            id=None,
            room_id=None,
            object_type=...,
            scale=chair.scale,
            relative_pose=chair.relative_pose,
            source_id=None,
        )

    position_argument = (
        table_position
        if table_position is not None
        else underspecified(EGPoint2D)(x=..., y=...)
    )
    scale_argument = (
        table_scale
        if table_scale is not None
        else underspecified(EGScale)(width=..., length=..., height=...)
    )
    orientation_argument = (
        table_orientation
        if table_orientation is not None
        else underspecified(EGRotation)(x=..., y=..., z=...)
    )
    return underspecified(EGTableWithChairs)(
        position=position_argument,
        scale=scale_argument,
        orientation=orientation_argument,
        chairs=[_fixed_slot(chair) for chair in fixed_chairs]
        + [_build_free_chair_query() for _ in range(free_count)],
    )


def build_free_table_query(chair_count: int):
    """
    Build a fully unconditioned EGTableWithChairs query with chair_count free
    chair slots.

    :param chair_count: Number of free chair slots to include in the
        query.
    :return: An underspecified EGTableWithChairs query with no fixed
        evidence.
    """
    return _build_conditioned_table_query([], chair_count)


def _fix_table_chairs(
    table_with_chairs: EGTableWithChairs,
    colliding_indices: set[int],
    rspn: RelationalProbabilisticCircuit,
) -> EGTableWithChairs:
    """
    Perform one repair pass: condition on non-colliding chairs and the table's
    own pose, and resample the given colliding indices.

    :param table_with_chairs: The table-with-chairs group to repair.
    :param colliding_indices: Indices into ``table_with_chairs.chairs``
        that must be resampled, as already computed by the caller.
    :param rspn: The fitted RSPN used to draw replacement chair poses.
    :return: A new EGTableWithChairs with colliding chairs replaced by
        fresh samples.
    """
    fixed_chairs = [
        chair
        for index, chair in enumerate(table_with_chairs.chairs)
        if index not in colliding_indices
    ]
    free_count = len(colliding_indices)
    query = _build_conditioned_table_query(
        fixed_chairs,
        free_count,
        table_position=table_with_chairs.position,
        table_scale=table_with_chairs.scale,
        table_orientation=table_with_chairs.orientation,
    )
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rspn)
    backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    new_sample = next(iter(backend.evaluate(query)))
    new_chairs = new_sample.chairs[len(fixed_chairs) :]
    return EGTableWithChairs(
        position=table_with_chairs.position,
        scale=table_with_chairs.scale,
        orientation=table_with_chairs.orientation,
        chairs=fixed_chairs + new_chairs,
        source_ids=table_with_chairs.source_ids,
    )


def resolve_table_chair_collisions(
    table_with_chairs: EGTableWithChairs,
    rspn: RelationalProbabilisticCircuit,
    max_passes: int = 50,
) -> EGTableWithChairs:
    """
    Return a collision-free version of table_with_chairs by iterating until no
    two chairs overlap.

    Only the chairs flagged in a given pass are resampled; already-clean
    chairs are fixed as conditioning evidence, mirroring
    :func:`~experiments.scene_generation_experiments.collision_resolution.resolve_shelf_collisions`.

    :param table_with_chairs: A table with sampled chairs around it.
    :param rspn: The fitted RSPN used to draw replacement chair poses.
    :param max_passes: Upper bound on repair passes before giving up.
    :raises TableChairLayoutResolutionError: If no collision-free
        arrangement is reached within *max_passes* repair passes.
    :return: An EGTableWithChairs with no pairwise chair collisions.
    """
    current = table_with_chairs
    colliding_indices: set[int] = set()
    for _ in range(max_passes):
        colliding_indices = _find_colliding_chair_indices(current)
        if not colliding_indices:
            return current
        current = _fix_table_chairs(current, colliding_indices, rspn)

    raise TableChairLayoutResolutionError(
        remaining_chair_indices=frozenset(colliding_indices),
        passes_attempted=max_passes,
    )
