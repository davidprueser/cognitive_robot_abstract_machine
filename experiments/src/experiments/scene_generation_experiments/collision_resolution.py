from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

from krrood.entity_query_language.factories import underspecified
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGRoomFloorLayout,
    EGShelfLayer,
    EGScale,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.room_floor_sampling import (
        SampledRoomShape,
    )


def minimal_resample_set(colliding_pairs: set[tuple[int, int]]) -> set[int]:
    """
    Return a minimal set of indices whose removal breaks every colliding pair.

    Greedy minimum vertex cover: repeatedly discard the index involved in the
    most remaining colliding pairs, breaking ties by the higher index for
    reproducibility. The result depends only on which indices collide, not on
    the order the pairs are reported, so callers get the same, stable choice
    regardless of how the underlying collision detector orders its contacts.

    :param colliding_pairs: Pairs of indices that collide, each a sorted
        ``(low, high)`` tuple.
    :return: Indices to resample so that no colliding pair remains.
    """
    remaining_pairs = set(colliding_pairs)
    indices_to_resample: set[int] = set()
    while remaining_pairs:
        involvement_counts = Counter(
            index for pair in remaining_pairs for index in pair
        )
        most_colliding_index = min(
            involvement_counts,
            key=lambda index: (-involvement_counts[index], -index),
        )
        indices_to_resample.add(most_colliding_index)
        remaining_pairs = {
            pair for pair in remaining_pairs if most_colliding_index not in pair
        }
    return indices_to_resample


def in_world_colliding_indices(
    detector: FCLCollisionDetector,
    bodies_by_index: dict[int, Body],
    static_obstacles: tuple[Body, ...] = (),
) -> set[int]:
    """
    Return a minimal set of *bodies_by_index* keys whose resampling clears every
    real-mesh collision among those bodies, and against *static_obstacles*, in
    the spawned world.

    Shared by the in-world shelf and table resolvers so both check collisions
    between real spawned bodies -- rather than box proxies -- through the same
    detector, matrix, and greedy-cover path. A body that hits a static
    obstacle (e.g. a shelf's corpus wall) is always resampled directly: unlike
    an inter-body collision, there is no choice of *which* side to move.

    :param detector: A collision detector already synced to the world the
        bodies live in; it re-syncs on the state changes body moves emit.
    :param bodies_by_index: The bodies to check against each other, keyed by
        their index in the owning collection.
    :param static_obstacles: Fixed bodies that never move, checked against
        each of *bodies_by_index* in addition to the pairwise checks among
        them.
    :return: Indices whose bodies must be resampled to remove all collisions.
    """
    body_to_index = {body: index for index, body in bodies_by_index.items()}
    collision_checks = {
        CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
        for body_a, body_b in combinations(body_to_index, 2)
    } | {
        CollisionCheck(body_a=body, body_b=obstacle, distance=0.0)
        for body in body_to_index
        for obstacle in static_obstacles
    }
    if not collision_checks:
        return set()
    result = detector.check_collisions(CollisionMatrix(collision_checks=collision_checks))
    if not result.any():
        return set()
    obstacles = set(static_obstacles)
    colliding_pairs: set[tuple[int, int]] = set()
    obstacle_hit_indices: set[int] = set()
    for contact in result.contacts:
        if contact.body_a in obstacles:
            obstacle_hit_indices.add(body_to_index[contact.body_b])
        elif contact.body_b in obstacles:
            obstacle_hit_indices.add(body_to_index[contact.body_a])
        else:
            colliding_pairs.add(
                tuple(sorted((body_to_index[contact.body_a], body_to_index[contact.body_b])))
            )
    return minimal_resample_set(colliding_pairs) | obstacle_hit_indices


def _build_free_object2d_query():
    """
    Build a fully underspecified EGObject2D query with all spatial fields free.

    Roll and pitch are fixed to ``0.0`` rather than left underspecified:
    floor objects always sit upright without tilting, so those two circuit
    dimensions are constant across every training example, and leaving a
    constant dimension underspecified lets the RSPN sampling backend leak
    the query's placeholder straight through instead of resolving it. Only
    yaw genuinely varies and is left for the RSPN to sample.

    :return: An underspecified EGObject2D with position, scale, and yaw
        unset.
    """
    return underspecified(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=...,
        scale=underspecified(EGScale)(width=..., length=..., height=...),
        position=underspecified(EGPoint2D)(x=..., y=...),
        orientation=underspecified(EGRotation)(x=0.0, y=0.0, z=...),
        source_id=None,
    )


def _fixed_object_slot(object_2d: EGObject2D):
    """
    Build an EGObject2D query slot whose spatial fields are pinned to
    *object_2d* as conditioning evidence.

    The object_type is left underspecified to avoid enum-to-float conversion
    issues in the RSPN sampling backend.

    :param object_2d: The object whose position, scale, and orientation are
        fixed.
    :return: A partially-underspecified EGObject2D holding *object_2d*'s pose.
    """
    return underspecified(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=...,
        scale=object_2d.scale,
        position=object_2d.position,
        orientation=object_2d.orientation,
        source_id=None,
    )


def _build_conditioned_layer_query(
    fixed_objects: list[EGObject2D],
    free_count: int,
    target_scale: EGScale | None = None,
):
    """
    Build an EGShelfLayer query conditioning on fixed_objects' spatial fields
    and leaving free_count slots fully underspecified.

    Each fixed object is represented as a partially-underspecified EGObject2D: position,
    scale, and orientation are fixed as literal values (conditioning evidence), while
    object_type is left underspecified to avoid enum-to-float conversion issues in the
    RSPN sampling backend.

    :param fixed_objects: Concrete EGObject2D instances whose spatial fields are fixed.
    :param free_count: Number of fully-underspecified object slots to resample.
    :param target_scale: When provided, the RSPN is conditioned on this scale so that
        sampled object positions are appropriate for the given layer dimensions.
        When ``None``, scale is sampled freely from the RSPN marginal.
    :return: An underspecified EGShelfLayer query ready for ProbabilisticBackend evaluation.
    """
    scale_argument = (
        target_scale
        if target_scale is not None
        else underspecified(EGScale)(width=..., length=..., height=...)
    )
    return build_pose_resample_query(fixed_objects, free_count, scale_argument)


def build_free_layer_query(object_count: int):
    """
    Build a fully unconditioned EGShelfLayer query with object_count free
    object slots.

    The layer scale is left free so the RSPN samples it from the
    marginal distribution. Use this to draw one reference layer whose
    scale can then be passed to
    :func:`build_layer_query_with_fixed_scale` for subsequent layers.

    :param object_count: Number of free object slots to include in the
        query.
    :return: An underspecified EGShelfLayer query with no fixed
        evidence.
    """
    return _build_conditioned_layer_query([], object_count)


def build_layer_query_with_fixed_scale(object_count: int, scale: EGScale):
    """
    Build an EGShelfLayer query with the layer scale fixed as conditioning
    evidence.

    The RSPN is conditioned on *scale* so that sampled object positions
    are drawn from the part of the learned distribution that is
    consistent with those dimensions. All layers of a shelf should be
    sampled with the same *scale* so the corpus can wrap them
    coherently.

    :param object_count: Number of free object slots to include in the
        query.
    :param scale: The target layer dimensions to condition on.
    :return: An underspecified EGShelfLayer query conditioned on
        *scale*.
    """
    return _build_conditioned_layer_query([], object_count, target_scale=scale)


def build_free_room_floor_query(shape: SampledRoomShape):
    """
    Build an :class:`EGRoomFloorLayout` query with the room's footprint fixed as
    conditioning evidence and ``shape.piece_count`` free floor-piece slots.

    The whole footprint is fixed rather than sampled, so the ``floor_area`` and
    ``aspect_ratio`` aggregations are *determined* by the query. Leaving width
    and length free would leave those latents undetermined, and grounding would
    integrate them out via Monte-Carlo -- which marginalises them out of the
    class circuit, statistically decoupling the footprint that ends up sampled
    from the one the piece positions were conditioned on, and multiplying
    grounding cost by the Monte-Carlo sample count.

    :param shape: The room's drawn piece count and floor footprint.
    :return: An underspecified :class:`EGRoomFloorLayout` query with the
        footprint as fixed evidence, ready for :class:`ProbabilisticBackend`
        evaluation.
    """
    return underspecified(EGRoomFloorLayout)(
        scale=shape.scale,
        pieces=[_build_free_object2d_query() for _ in range(shape.piece_count)],
    )


def build_pose_resample_query(
    fixed_objects: list[EGObject2D],
    free_count: int,
    layer_scale: EGScale,
):
    """
    Build an EGShelfLayer query that keeps every non-resampled object fixed and
    redraws the scale, position, and orientation of free_count fresh objects.

    Conditioning a resampled slot on its own scale, in addition to the other
    objects' exact poses, pins the query to the single training example that
    combination of evidence came from, collapsing the RSPN's posterior for
    that slot's position back to its original, still-colliding value -- so the
    scale is left free like every other field. The caller keeps the body's
    existing mesh regardless, since it only ever applies the redrawn position
    and orientation, never the redrawn scale. Free slots are appended after
    the fixed ones, so the caller reads the redrawn objects off the tail of
    the result.

    :param fixed_objects: Objects whose full pose is held as evidence.
    :param free_count: Number of fully-underspecified object slots to
        resample.
    :param layer_scale: The layer dimensions to condition on.
    :return: An underspecified EGShelfLayer query ready for
        :class:`ProbabilisticBackend` evaluation.
    """
    return underspecified(EGShelfLayer)(
        scale=layer_scale,
        objects=[_fixed_object_slot(object_2d) for object_2d in fixed_objects]
        + [_build_free_object2d_query() for _ in range(free_count)],
    )
