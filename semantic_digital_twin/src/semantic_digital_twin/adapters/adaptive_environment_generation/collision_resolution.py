from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Set, Tuple

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.adapters.adaptive_environment_generation.schema import (
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


def _create_book_world(layer: EGShelfLayer) -> Tuple[World, Dict[Body, int]]:
    """
    Build a temporary world with one box body per EGObject2D in the layer.

    Each body uses the object's scale as its box extent and the object's
    2-D position and orientation as its placement.

    :param layer: The shelf layer whose objects should be represented.
    :return: Tuple of (world, body_to_index) where body_to_index maps each
        created Body back to its index in layer.objects.
    """
    world = World()
    root = Body(name=PrefixedName(name="collision_root"))
    body_to_index: Dict[Body, int] = {}

    with world.modify_world():
        for i, obj in enumerate(layer.objects):
            if not isinstance(obj.position.x, (int, float)):
                continue
            body = Body(
                name=PrefixedName(name=f"book_{i}"),
                collision=ShapeCollection(
                    [
                        Box(
                            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
                            scale=Scale(
                                x=obj.scale.width,
                                y=obj.scale.length,
                                z=obj.scale.height,
                            ),
                        )
                    ]
                ),
            )
            conn = Connection6DoF.create_with_dofs(parent=root, child=body, world=world)
            world.add_body(body)
            world.add_connection(conn)
            body_to_index[body] = i

    for body, i in body_to_index.items():
        obj = layer.objects[i]
        body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            obj.position.x,
            obj.position.y,
            0.0,
            *obj.orientation.as_roll_pitch_yaw_in_radians(),
        )

    return world, body_to_index


def _find_colliding_indices(layer: EGShelfLayer) -> Set[int]:
    """
    Return the minimal set of object indices that must be resampled to resolve all collisions.

    For each colliding pair, only one index is added to the bad set (greedy: keep the
    first, discard the second).  If one member of a pair is already in the bad set, the
    pair is already handled and the other member is kept.

    :param layer: The shelf layer to inspect.
    :return: Set of indices (into layer.objects) that must be replaced.
    """
    world, body_to_index = _create_book_world(layer)
    if len(body_to_index) < 2:
        return set()

    detector = FCLCollisionDetector(_world=world)
    collision_matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=a, body_b=b, distance=0.0)
            for a, b in combinations(body_to_index.keys(), 2)
        }
    )
    result = detector.check_collisions(collision_matrix)
    if not result.any():
        return set()

    pairs: List[Tuple[int, int]] = [
        (body_to_index[c.body_a], body_to_index[c.body_b])
        for c in result.contacts
    ]

    bad: Set[int] = set()
    for i, j in pairs:
        if i not in bad and j not in bad:
            bad.add(j)
    return bad


def _build_conditioned_layer_query(good_objects: List[EGObject2D], bad_count: int):
    """
    Build an EGShelfLayer query conditioning on good_objects' spatial fields and leaving
    bad_count slots fully underspecified.

    Each good object is represented as a partially-underspecified EGObject2D: position,
    scale, and orientation are fixed as literal values (conditioning evidence), while
    object_type is left underspecified to avoid enum-to-float conversion issues in the
    RSPN sampling backend.

    :param good_objects: Concrete EGObject2D instances whose spatial fields are fixed.
    :param bad_count: Number of fully-underspecified object slots to resample.
    :return: An underspecified EGShelfLayer query ready for ProbabilisticBackend evaluation.
    """

    def _good_slot(obj: EGObject2D):
        return underspecified(EGObject2D)(
            id=None,
            room_id=None,
            place_id=None,
            object_type=...,
            scale=obj.scale,
            position=obj.position,
            orientation=obj.orientation,
            source_id=None,
        )

    return underspecified(EGShelfLayer)(
        scale=underspecified(EGSize)(width=..., length=..., height=...),
        objects=[_good_slot(obj) for obj in good_objects]
        + [
            underspecified(EGObject2D)(
                id=None,
                room_id=None,
                place_id=None,
                object_type=...,
                scale=underspecified(EGSize)(width=..., length=..., height=...),
                position=underspecified(EGPoint2D)(x=..., y=...),
                orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
                source_id=None,
            )
            for _ in range(bad_count)
        ],
    )


def _fix_layer(
    layer: EGShelfLayer,
    rspn: RelationalProbabilisticCircuit,
) -> EGShelfLayer:
    """
    Perform one repair pass on a layer: detect collisions, condition on valid books,
    and resample the minimal set of colliding books.

    :param layer: The shelf layer to repair.
    :param rspn: The fitted RSPN used to draw replacement book positions.
    :return: A new EGShelfLayer with colliding books replaced by fresh samples.
    """
    bad = _find_colliding_indices(layer)
    if not bad:
        return layer
    good_objects = [obj for i, obj in enumerate(layer.objects) if i not in bad]
    bad_count = len(bad)
    query = _build_conditioned_layer_query(good_objects, bad_count)
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rspn, query=query
    )
    backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    new_layer = next(iter(backend.evaluate(query)))
    # Restore original good objects (preserves object_type left free in conditioned slots);
    # take the trailing bad_count entries as the newly sampled replacements.
    new_bad_objects = new_layer.objects[len(good_objects):]
    return EGShelfLayer(scale=layer.scale, objects=good_objects + new_bad_objects)


def resolve_shelf_collisions(
    layers: List[EGShelfLayer],
    rspn: RelationalProbabilisticCircuit,
) -> List[EGShelfLayer]:
    """
    Return collision-free versions of all shelf layers by iterating until every layer
    is clean.

    The outer loop repeats until no layer contains any colliding book pair.  On each
    pass only layers that still have collisions are repaired, so already-clean layers
    are never touched again.

    :param layers: All layers of a shelf, each containing sampled EGObject2D books.
    :param rspn: The fitted RSPN used to draw replacement book positions.
    :return: A list of EGShelfLayer instances with no pairwise book collisions.
    """
    layers = list(layers)
    while True:
        dirty_indices = [
            i for i, layer in enumerate(layers) if _find_colliding_indices(layer)
        ]
        if not dirty_indices:
            return layers
        for i in dirty_indices:
            layers[i] = _fix_layer(layers[i], rspn)
