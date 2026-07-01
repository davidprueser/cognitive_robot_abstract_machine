from __future__ import annotations

from itertools import combinations

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGOrientation,
    EGPoint2D,
    EGShelfLayer,
    EGSize,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _create_book_world(layer: EGShelfLayer) -> tuple[World, dict[Body, int]]:
    """
    Build a temporary world with one box body per EGObject2D in the layer.

    Each body uses the object's scale as its box extent and the object's
    2-D position and orientation as its placement.

    :param layer: The shelf layer whose objects should be represented.
    :return: Tuple of (world, body_to_index) where body_to_index maps
        each created Body back to its index in layer.objects.
    """
    world = World()
    root = Body(name=PrefixedName(name="collision_root"))
    body_to_index: dict[Body, int] = {}

    with world.modify_world():
        for index, object_2d in enumerate(layer.objects):
            if not isinstance(object_2d.position.x, (int, float)):
                continue
            body = Body(
                name=PrefixedName(name=f"book_{index}"),
                collision=ShapeCollection(
                    [
                        Box(
                            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
                            scale=Scale(
                                x=object_2d.scale.width,
                                y=object_2d.scale.length,
                                z=object_2d.scale.height,
                            ),
                        )
                    ]
                ),
            )
            conn = Connection6DoF.create_with_dofs(parent=root, child=body, world=world)
            world.add_body(body)
            world.add_connection(conn)
            body_to_index[body] = index

    for body, index in body_to_index.items():
        object_2d = layer.objects[index]
        body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            object_2d.position.x,
            object_2d.position.y,
            0.0,
            *object_2d.orientation.as_roll_pitch_yaw_in_radians(),
        )

    return world, body_to_index


def _find_colliding_indices(layer: EGShelfLayer) -> set[int]:
    """
    Return the minimal set of object indices that must be resampled to resolve
    all collisions.

    For each colliding pair, only one index is added to the bad set
    (greedy: keep the first, discard the second).  If one member of a
    pair is already in the bad set, the pair is already handled and the
    other member is kept.

    :param layer: The shelf layer to inspect.
    :return: Set of indices (into layer.objects) that must be replaced.
    """
    world, body_to_index = _create_book_world(layer)
    if len(body_to_index) < 2:
        return set()

    detector = FCLCollisionDetector(_world=world)
    collision_matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(body_to_index.keys(), 2)
        }
    )
    result = detector.check_collisions(collision_matrix)
    if not result.any():
        return set()

    colliding_pairs: list[tuple[int, int]] = [
        (body_to_index[contact.body_a], body_to_index[contact.body_b])
        for contact in result.contacts
    ]

    indices_to_resample: set[int] = set()
    for first_index, second_index in colliding_pairs:
        if first_index not in indices_to_resample and second_index not in indices_to_resample:
            indices_to_resample.add(second_index)
    return indices_to_resample


def _build_free_object2d_query():
    """
    Build a fully underspecified EGObject2D query with all spatial fields free.

    :return: An underspecified EGObject2D with position, scale, and
        orientation unset.
    """
    return underspecified(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=...,
        scale=underspecified(EGSize)(width=..., length=..., height=...),
        position=underspecified(EGPoint2D)(x=..., y=...),
        orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
        source_id=None,
    )


def _build_conditioned_layer_query(
    fixed_objects: list[EGObject2D],
    free_count: int,
    target_scale: EGSize | None = None,
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

    def _fixed_slot(object_2d: EGObject2D):
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

    scale_argument = (
        target_scale
        if target_scale is not None
        else underspecified(EGSize)(width=..., length=..., height=...)
    )
    return underspecified(EGShelfLayer)(
        scale=scale_argument,
        objects=[_fixed_slot(object_2d) for object_2d in fixed_objects]
        + [_build_free_object2d_query() for _ in range(free_count)],
    )


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


def build_layer_query_with_fixed_scale(object_count: int, scale: EGSize):
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


def _fix_layer(
    layer: EGShelfLayer,
    colliding_indices: set[int],
    rspn: RelationalProbabilisticCircuit,
) -> EGShelfLayer:
    """
    Perform one repair pass on a layer: condition on valid books and resample
    the given colliding indices.

    :param layer: The shelf layer to repair.
    :param colliding_indices: Indices into ``layer.objects`` that must
        be resampled, as already computed by the caller.
    :param rspn: The fitted RSPN used to draw replacement book
        positions.
    :return: A new EGShelfLayer with colliding books replaced by fresh
        samples.
    """
    fixed_objects = [object_2d for index, object_2d in enumerate(layer.objects) if index not in colliding_indices]
    free_count = len(colliding_indices)
    query = _build_conditioned_layer_query(fixed_objects, free_count, target_scale=layer.scale)
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rspn
    )
    backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    new_layer = next(iter(backend.evaluate(query)))
    # Restore original fixed objects (preserves object_type left free in conditioned slots);
    # take the trailing free_count entries as the newly sampled replacements.
    new_objects = new_layer.objects[len(fixed_objects):]
    return EGShelfLayer(scale=layer.scale, objects=fixed_objects + new_objects)


def resolve_shelf_collisions(
    layers: list[EGShelfLayer],
    rspn: RelationalProbabilisticCircuit,
) -> list[EGShelfLayer]:
    """
    Return collision-free versions of all shelf layers by iterating until every
    layer is clean.

    The outer loop repeats until no layer contains any colliding book
    pair.  On each pass only layers that still have collisions are
    repaired, so already-clean layers are never touched again. The
    colliding indices found while deciding whether a layer needs repair
    are reused for the repair itself, instead of being recomputed.

    :param layers: All layers of a shelf, each containing sampled
        EGObject2D books.
    :param rspn: The fitted RSPN used to draw replacement book
        positions.
    :return: A list of EGShelfLayer instances with no pairwise book
        collisions.
    """
    layers = list(layers)
    while True:
        colliding_indices_by_layer = {
            index: colliding_indices
            for index, layer in enumerate(layers)
            if (colliding_indices := _find_colliding_indices(layer))
        }
        if not colliding_indices_by_layer:
            return layers
        for index, colliding_indices in colliding_indices_by_layer.items():
            layers[index] = _fix_layer(layers[index], colliding_indices, rspn)