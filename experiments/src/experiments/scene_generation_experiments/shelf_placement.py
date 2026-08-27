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
    EGPoint2D,
    EGRotation,
    EGShelfLayer,
    SpawnedShelf,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import ShelfLayer
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    translate_free_space_to_where_condition,
)


def mode_query(
    spawned_shelf: SpawnedShelf,
    shelf_circuit: RelationalProbabilisticCircuit,
    held_object: EGObject2D,
) -> tuple[EGObject2D, str]:
    """
    Ask the fitted shelf model where *held_object* most likely belongs on
    *spawned_shelf*.

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

    :param spawned_shelf: The shelf, already standing in a world, to place onto.
    :param shelf_circuit: The fitted circuit rooted at :class:`EGShelf`.
    :param held_object: The object to place; its type and size are what the model is
        asked about, its position and orientation are what the answer replaces.
    :raises NoShelfPlacementError: If no layer has room for the object.
    :return: The placed object, and the name of the real layer
        (:attr:`~semantic_digital_twin.scene_generation.scene_schema.EGShelfLayer.
        annotated_layer`'s root body) it was placed on.
    """
    layer_circuit = shelf_circuit.exchangeable_distribution_templates[
        "layers"
    ].template_distribution
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=layer_circuit)
    backend = ProbabilisticBackend(model_registry=registry)

    candidates: list[tuple[float, EGShelfLayer, EGShelfLayer]] = []
    for layer in spawned_shelf.shelf.layers:
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
            shelf_name=str(spawned_shelf.corpus.name),
            object_type=held_object.object_type.value,
        )

    _, layer, placed_layer = max(candidates, key=lambda candidate: candidate[0])
    return placed_layer.objects[-1], str(layer.annotated_layer.root.name)


def layer_named(spawned_shelf: SpawnedShelf, layer_name: str) -> EGShelfLayer:
    """
    Resolve the layer :func:`mode_query` placed onto, from the name it returned
    alongside the placed object.

    :param spawned_shelf: The shelf the layer belongs to.
    :param layer_name: The layer name :func:`mode_query` returned.
    :raises StopIteration: If no layer of *spawned_shelf* carries that name.
    :return: The matching layer.
    """
    return next(
        layer
        for layer in spawned_shelf.shelf.layers
        if str(layer.annotated_layer.root.name) == layer_name
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
    :param held_object: The object whose type and size are pinned, and whose position
        and yaw are left for the model to answer.
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
    query.where(
        _free_space_where_condition(
            layer.annotated_layer, held_slot.variable.position, held_object
        )
    )
    return query


def _held_object_slot(held_object: EGObject2D):
    """
    Build a query slot for *held_object*, pinning its type and size and leaving its
    position and yaw for the model to answer.

    Roll and pitch are pinned upright, as everywhere else the model is asked; only yaw
    genuinely varies.

    :param held_object: The object whose type and size are pinned.
    :return: An underspecified EGObject2D slot.
    """
    return a(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=held_object.object_type,
        scale=held_object.scale,
        position=a(EGPoint2D)(x=..., y=...),
        orientation=a(EGRotation)(x=0.0, y=0.0, z=...),
        source_id=None,
    )


def _free_space_where_condition(
    layer_annotation: ShelfLayer, position: EQLVariable, held_object: EGObject2D
) -> OR:
    """
    Build the where-condition keeping *position* inside *layer_annotation*'s own free
    space.

    The free space is bloated by *held_object*'s own circumradius before *position* is
    constrained to it: the where-condition only pins *position*, never the yaw the
    model separately answers for the same slot, so a bloat that only covered the
    object's half-width would still let a corner reach into a neighbour once the model
    picked a diagonal yaw. The circumradius covers the object's reach at every yaw.

    :param layer_annotation: The layer whose free space bounds *position*. Its
        supporting surface is expected to already be calculated -- as
        :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
        spawn_in_world` does for every layer it spawns -- and, if the shelf was merged
        into another world since, to already be live in it -- as
        :meth:`~semantic_digital_twin.scene_generation.scene_schema.SpawnedShelf.
        refresh_layer_annotations` makes it.
    :param position: The query variable to constrain.
    :param held_object: The object *position* will place; its footprint decides how
        much the free space is bloated by.
    :return: The where condition restricting *position* to the layer's free space.
    """
    object_bloat = max(held_object.scale.width, held_object.scale.length) / 2
    free_space = layer_annotation.calculate_free_space(object_bloat=object_bloat)
    return translate_free_space_to_where_condition(
        free_space.free_space_event,
        position,
    )
