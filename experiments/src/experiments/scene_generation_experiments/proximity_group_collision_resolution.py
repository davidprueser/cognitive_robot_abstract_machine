from __future__ import annotations

import random

from krrood.entity_query_language.factories import underspecified
from semantic_digital_twin.scene_generation.scene_schema import (
    EGGroupMember,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGProximityGroup,
    ObjectType,
)


def sample_member_count(
    member_counts_by_anchor_type: dict[ObjectType, list[int]],
    anchor_type: ObjectType,
) -> int:
    """
    Draw how many members to place around an anchor of *anchor_type*, from the
    counts observed around anchors of that same type.

    The count has to be drawn before the sampling query is built, since an
    exchangeable relation's list length is a structural property of the query,
    not something ``ground()`` can pick on its own. Drawing from the empirical
    distribution and then conditioning the query on that count (via the
    ``total_count`` aggregation statistic) is equivalent to
    ``P(count) * P(scale, poses | count)`` -- the same joint the RSPN learned,
    just factored so that sampling ``count`` does not depend on marginalising
    the fitted circuit down to a single aggregation variable, which the
    underlying JPT does not support for aggregation statistics.

    Conditioning on the anchor's type is what keeps solitary furniture solitary:
    the fitted circuit is one model over every kind of group, so it cannot be
    asked for "members appropriate to a refrigerator". The count carries that
    conditioning instead, and a type whose training anchors stand alone draws
    zero.

    :param member_counts_by_anchor_type: Member counts observed per anchor
        type, from
        :func:`~experiments.scene_generation_experiments.proximity_group_generation.member_counts_by_anchor_type`.
    :param anchor_type: The category of the anchor being furnished.
    :return: A member count drawn from that type's training distribution, or
        zero for a type never seen anchoring a group.
    """
    observed_counts = member_counts_by_anchor_type.get(anchor_type)
    if not observed_counts:
        return 0
    return random.choice(observed_counts)


def _build_free_member_query():
    """
    Build a fully underspecified EGGroupMember query with all pose fields free.

    :return: An underspecified EGGroupMember with scale and relative_pose
        unset.
    """
    return underspecified(EGGroupMember)(
        id=None,
        room_id=None,
        object_type=...,
        scale=underspecified(EGScale)(width=..., length=..., height=...),
        relative_pose=underspecified(EGRelativePolarPose)(
            distance_from_anchor=...,
            angle_from_anchor=...,
            facing_angle_relative_to_anchor=...,
        ),
        source_id=None,
    )


def _fixed_member_slot(member: EGGroupMember):
    """
    Build an EGGroupMember query slot whose scale and relative pose are pinned to
    *member* as conditioning evidence.

    The object_type is left underspecified to avoid enum-to-float conversion
    issues in the RSPN sampling backend.

    :param member: The member whose scale and relative pose are fixed.
    :return: A partially-underspecified EGGroupMember holding *member*'s pose.
    """
    return underspecified(EGGroupMember)(
        id=None,
        room_id=None,
        object_type=...,
        scale=member.scale,
        relative_pose=member.relative_pose,
        source_id=None,
    )


def _build_conditioned_group_query(
    fixed_members: list[EGGroupMember],
    free_count: int,
    anchor_position: EGPoint2D | None = None,
    anchor_scale: EGScale | None = None,
    anchor_orientation: EGRotation | None = None,
):
    """
    Build an EGProximityGroup query conditioning on fixed_members' poses and
    leaving free_count member slots fully underspecified.

    :param fixed_members: Concrete EGGroupMember instances whose relative pose
        is fixed as conditioning evidence.
    :param free_count: Number of fully-underspecified member slots to
        resample.
    :param anchor_position: When provided, fixes the anchor's own position
        as conditioning evidence, so a repair pass samples members for
        the same anchor instance rather than an entirely new one.
    :param anchor_scale: When provided, fixes the anchor's own scale as
        conditioning evidence.
    :param anchor_orientation: When provided, fixes the anchor's own
        orientation as conditioning evidence.
    :return: An underspecified EGProximityGroup query ready for
        ProbabilisticBackend evaluation.
    """
    position_argument = (
        anchor_position
        if anchor_position is not None
        else underspecified(EGPoint2D)(x=..., y=...)
    )
    scale_argument = (
        anchor_scale
        if anchor_scale is not None
        else underspecified(EGScale)(width=..., length=..., height=...)
    )
    orientation_argument = (
        anchor_orientation
        if anchor_orientation is not None
        else underspecified(EGRotation)(x=..., y=..., z=...)
    )
    return underspecified(EGProximityGroup)(
        position=position_argument,
        scale=scale_argument,
        orientation=orientation_argument,
        members=[_fixed_member_slot(member) for member in fixed_members]
        + [_build_free_member_query() for _ in range(free_count)],
    )


def build_member_pose_resample_query(
    fixed_members: list[EGGroupMember],
    resampled_members: list[EGGroupMember],
    anchor_scale: EGScale,
):
    """
    Build an EGProximityGroup query that keeps every non-resampled member fixed
    and redraws the scale and relative pose of the resampled members.

    Conditioning a resampled slot on its own scale, in addition to the other
    members' exact poses, pins the query to the single training example that
    combination of evidence came from, collapsing the RSPN's posterior for
    that slot's relative pose back to its original, still-colliding value --
    so the scale is left free like every other field. The caller keeps the
    body's existing mesh regardless, since it only ever applies the redrawn
    relative pose, never the redrawn scale. Resampled slots are appended
    after the fixed ones, so the caller reads the redrawn members off the tail
    of the result in the order of *resampled_members*.

    Only the anchor's scale is held as evidence. A member's pose is polar and
    relative to its anchor, with the anchor's yaw already subtracted by
    :meth:`EGRelativePolarPose.from_absolute_poses`, so the anchor's absolute
    position and orientation carry no information about it and pinning them
    only shrinks the circuit's support. Pinning the position was in fact fatal:
    repair passes supply the anchor's room-centred position while the circuit is
    fitted on raw room coordinates, so any anchor left of the room centre
    conditioned on a negative coordinate of zero probability mass and aborted
    the layout with ``NoSolutionFound``.

    :param fixed_members: Chairs whose full pose is held as evidence.
    :param resampled_members: Chairs whose relative pose is redrawn; only used
        to determine how many free slots to add.
    :param anchor_scale: The anchor scale to condition on.
    :return: An underspecified EGProximityGroup query ready for
        :class:`ProbabilisticBackend` evaluation.
    """
    return underspecified(EGProximityGroup)(
        position=underspecified(EGPoint2D)(x=..., y=...),
        scale=anchor_scale,
        orientation=underspecified(EGRotation)(x=..., y=..., z=...),
        members=[_fixed_member_slot(member) for member in fixed_members]
        + [_build_free_member_query() for _ in resampled_members],
    )


def build_free_group_query(member_count: int):
    """
    Build a fully unconditioned EGProximityGroup query with member_count free
    member slots.

    :param member_count: Number of free member slots to include in the
        query.
    :return: An underspecified EGProximityGroup query with no fixed
        evidence.
    """
    return _build_conditioned_group_query([], member_count)


