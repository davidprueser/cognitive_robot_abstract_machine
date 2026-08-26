from __future__ import annotations

import math
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.core.variable import Variable as EQLVariable
from krrood.entity_query_language.operators.core_logical_operators import OR
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
    EGScale,
    EGShelf,
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
) -> EGShelf:
    """
    Ask the fitted shelf model where *held_object* most likely belongs on
    *spawned_shelf*.

    Every layer is asked separately, for a layer holding what it already holds plus
    one more object of the held object's type and size, and the layer and pose with
    the highest joint density win. A layer that has no room above its slab for the
    object, or no space left on it, is never asked.

    .. note::
        What the object is decides *where on a layer* it goes, not which layer. A
        fitted circuit passes only aggregation statistics from a layer to its
        objects, so the object's type and size never reach the layer's own height --
        the layers are separated by how typical their height and their pose density
        are, and no amount of evidence about the object changes that.

    :param spawned_shelf: The shelf, already standing in a world, to place onto.
    :param shelf_circuit: The fitted circuit rooted at :class:`EGShelf`.
    :param held_object: The object to place; its type and size are what the model is
        asked about, its position and orientation are what the answer replaces.
    :param sample_count: Poses drawn per layer to search for the densest one.
    :raises NoShelfPlacementError: If no layer can take the object.
    :return: The most likely placement across all layers.
    """
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=shelf_circuit)
    query = _shelf_query(spawned_shelf.shelf, held_object)
    backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    sample = next(iter(backend.evaluate_mode(query)))
    return sample


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


def _shelf_query(shelf: EGShelf, held_object: EGObject2D):
    """
    Build an EGShelf query holding every layer's own objects plus one held-object slot
    on each layer, asking where across the whole shelf the object is most likely to go.

    Each layer gets its own held-object slot, built fresh rather than shared: a query
    is a tree, and a slot placed under more than one layer at once would leave the
    layers unable to tell their own copy of it apart by name.

    :param shelf: The shelf to hold the query on, with every layer's own objects as
        fixed evidence.
    :param held_object: The object whose type and size are pinned, and whose position
        and yaw are left for the model to answer, on every layer's own slot.
    :return: An underspecified EGShelf query whose every layer's last object slot is a
        held one.
    """
    held_slots = [_held_object_slot(held_object) for _ in shelf.layers]

    query = a(EGShelf)(
        scale=a(EGScale)(width=..., length=..., height=...),
        layers=[
            a(EGShelfLayer)(
                objects=[_fixed_object_slot(obj) for obj in layer.objects]
                + [held_slot],
                theme_dominant_type=layer.theme_dominant_type,
                height_above_shelf_base=layer.height_above_shelf_base,
                relative_height=layer.relative_height,
                vertical_clearance=layer.vertical_clearance,
            )
            for layer, held_slot in zip(shelf.layers, held_slots)
        ],
        theme_dominant_type=shelf.theme_dominant_type,
    )
    for layer, held_slot in zip(shelf.layers, held_slots):
        query.where(
            _free_space_where_condition(
                layer.annotated_layer, held_slot.variable.position, held_object
            )
        )
    return query
