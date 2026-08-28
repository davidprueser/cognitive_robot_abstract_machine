from __future__ import annotations

import math
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.core.variable import Variable as EQLVariable
from krrood.entity_query_language.exceptions import NoSolutionFound
from krrood.entity_query_language.operators.core_logical_operators import OR
from experiments.scene_generation_experiments.exceptions import NoShelfPlacementError
from experiments.scene_generation_experiments.rspn_sampling import (
    _fixed_object_slot,
)
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import ShelfLayer
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    translate_free_space_to_where_condition,
)


def mode_query(
    shelf: EGShelf,
    shelf_circuit: RelationalProbabilisticCircuit,
    held_object: EGObject2D,
) -> tuple[EGObject2D, str]:
    """
    Ask the fitted shelf model where *held_object* most likely belongs on *shelf*.

    Every layer is asked on its own -- the narrow, per-layer circuit a mode query can
    answer exactly, rather than the shelf's whole joint circuit -- and the layer and
    pose with the highest absolute log-density wins. A layer with no free space for the
    object raises :class:`~krrood.entity_query_language.exceptions.NoSolutionFound`
    while it is being asked and is skipped rather than propagating that error.

    .. note::
        What the object is decides *where on a layer* it goes, not *which* layer. A
        fitted circuit passes only aggregation statistics from a layer to its
        objects, so the object's type and size never reach the layer's own height --
        the layers are separated by how typical their own attributes and free space
        are, not by what is being placed on them.

    :param shelf: The shelf, already standing in a world, to place onto.
    :param shelf_circuit: The fitted circuit rooted at :class:`EGShelf`.
    :param held_object: The object to place; its type and size are what the model is
        asked about, its pose is what the answer replaces.
    :raises NoShelfPlacementError: If no layer has room for the object.
    :return: The placed object, and the name of the real layer
        (:attr:`~semantic_digital_twin.scene_generation.scene_schema.EGShelfLayer.
        annotation`'s root body) it was placed on.
    """
    layer_circuit = shelf_circuit.exchangeable_distribution_templates[
        "layers"
    ].template_distribution
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=layer_circuit)
    backend = ProbabilisticBackend(model_registry=registry)

    candidates: list[tuple[float, EGShelfLayer, EGShelfLayer]] = []
    for layer in shelf.layers:
        try:
            placed_layer, log_density = next(
                iter(
                    backend.evaluate_mode_with_log_density(
                        _layer_query(layer, held_object)
                    )
                )
            )
        except NoSolutionFound:
            continue
        candidates.append((log_density, layer, placed_layer))

    if not candidates:
        raise NoShelfPlacementError(
            shelf_name=str(shelf.corpus.name),
            object_type=held_object.object_type.value,
        )

    _, layer, placed_layer = max(candidates, key=lambda candidate: candidate[0])
    return placed_layer.objects[-1], str(layer.annotation.root.name)


def layer_named(shelf: EGShelf, layer_name: str) -> EGShelfLayer:
    """
    Resolve the layer :func:`mode_query` placed onto, from the name it returned
    alongside the placed object.

    :param shelf: The shelf the layer belongs to.
    :param layer_name: The layer name :func:`mode_query` returned.
    :raises StopIteration: If no layer of *shelf* carries that name.
    :return: The matching layer.
    """
    return next(
        layer for layer in shelf.layers if str(layer.annotation.root.name) == layer_name
    )


def _layer_query(layer: EGShelfLayer, held_object: EGObject2D):
    """
    Build a standalone :class:`EGShelfLayer` query for placing *held_object* onto
    *layer*.

    Rooted at the layer itself, not nested under its shelf, so it grounds against the
    narrow per-layer circuit that a mode query can answer exactly (see
    :func:`mode_query`).

    :param layer: The layer to place the object onto; its existing objects and its own
        attributes are held as fixed evidence.
    :param held_object: The object whose type and size are pinned, and whose pose is
        left for the model to answer.
    :return: An underspecified EGShelfLayer query whose last object slot is the held
        one.
    """
    held_slot = _held_object_slot(held_object)
    query = a(EGShelfLayer)(
        objects=[_fixed_object_slot(obj) for obj in layer.objects] + [held_slot],
        theme_dominant_type=layer.theme_dominant_type,
        height_above_shelf_base=layer.height_above_shelf_base,
        relative_height=layer.relative_height,
        vertical_clearance=layer.vertical_clearance,
    )
    # Resolve first: held_slot.variable is only reassigned to its "objects[N]" access-path
    # variable once query resolves its objects list, and held_slot.variable.pose below
    # must build on that variable, not a disconnected one truncation could never match.
    query.resolve()
    query.where(
        _free_space_where_condition(
            layer.annotation, held_slot.variable.pose, held_object
        )
    )
    return query


def _held_object_slot(held_object: EGObject2D):
    """
    Build a query slot for *held_object*, pinning its type and size and leaving its pose
    for the model to answer.

    :param held_object: The object whose type and size are pinned.
    :return: An underspecified EGObject2D slot.
    """
    return a(EGObject2D)(
        object_type=held_object.object_type,
        scale=held_object.scale,
        pose=a(Pose2D)(x=..., y=..., yaw=...),
        source_id=None,
    )


def _free_space_where_condition(
    layer_annotation: ShelfLayer, pose: EQLVariable, held_object: EGObject2D
) -> OR:
    """
    Build the where-condition keeping *pose* inside *layer_annotation*'s own free space.

    The free space is bloated by *held_object*'s own circumradius before *pose* is
    constrained to it: the where-condition only pins *pose*'s x/y, never the yaw the
    model separately answers for the same slot, so a bloat that only covered the
    object's half-width would still let a corner reach into a neighbour once the model
    picked a diagonal yaw. The circumradius covers the object's reach at every yaw.

    :param layer_annotation: The layer whose free space bounds *pose*. Its
        supporting surface is expected to already be calculated -- as
        :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
        spawn` does for every layer it spawns -- and, if the shelf was merged
        into another world since, to already be live in it -- as
        :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
        refresh_layer_annotations` makes it.
    :param pose: The query variable to constrain.
    :param held_object: The object *pose* will place; its footprint decides how
        much the free space is bloated by.
    :return: The where condition restricting *pose* to the layer's free space.
    """
    object_bloat = max(held_object.scale.y, held_object.scale.x) / 2
    free_space = layer_annotation.calculate_free_space(object_bloat=object_bloat)
    return translate_free_space_to_where_condition(
        free_space.free_space_event,
        pose,
    )
