from __future__ import annotations

import random

from krrood.entity_query_language.factories import underspecified
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
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


def _fixed_chair_slot(chair: EGChair):
    """
    Build an EGChair query slot whose scale and relative pose are pinned to
    *chair* as conditioning evidence.

    The object_type is left underspecified to avoid enum-to-float conversion
    issues in the RSPN sampling backend.

    :param chair: The chair whose scale and relative pose are fixed.
    :return: A partially-underspecified EGChair holding *chair*'s pose.
    """
    return underspecified(EGChair)(
        id=None,
        room_id=None,
        object_type=...,
        scale=chair.scale,
        relative_pose=chair.relative_pose,
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
        chairs=[_fixed_chair_slot(chair) for chair in fixed_chairs]
        + [_build_free_chair_query() for _ in range(free_count)],
    )


def build_chair_pose_resample_query(
    fixed_chairs: list[EGChair],
    resampled_chairs: list[EGChair],
    table_position: EGPoint2D,
    table_scale: EGScale,
    table_orientation: EGRotation,
):
    """
    Build an EGTableWithChairs query that keeps every non-resampled chair fixed
    and redraws the scale and relative pose of the resampled chairs.

    Conditioning a resampled slot on its own scale, in addition to the other
    chairs' exact poses, pins the query to the single training example that
    combination of evidence came from, collapsing the RSPN's posterior for
    that slot's relative pose back to its original, still-colliding value --
    so the scale is left free like every other field. The caller keeps the
    body's existing mesh regardless, since it only ever applies the redrawn
    relative pose, never the redrawn scale. Resampled slots are appended
    after the fixed ones, so the caller reads the redrawn chairs off the tail
    of the result in the order of *resampled_chairs*.

    :param fixed_chairs: Chairs whose full pose is held as evidence.
    :param resampled_chairs: Chairs whose relative pose is redrawn; only used
        to determine how many free slots to add.
    :param table_position: The table position to condition on.
    :param table_scale: The table scale to condition on.
    :param table_orientation: The table orientation to condition on.
    :return: An underspecified EGTableWithChairs query ready for
        :class:`ProbabilisticBackend` evaluation.
    """
    return underspecified(EGTableWithChairs)(
        position=table_position,
        scale=table_scale,
        orientation=table_orientation,
        chairs=[_fixed_chair_slot(chair) for chair in fixed_chairs]
        + [_build_free_chair_query() for _ in resampled_chairs],
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


