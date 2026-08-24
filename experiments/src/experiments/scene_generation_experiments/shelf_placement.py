from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import numpy as np
from random_events.interval import Interval, closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from typing_extensions import Any

from experiments.scene_generation_experiments.exceptions import NoShelfPlacementError
from experiments.scene_generation_experiments.rspn_sampling import free_object_slot
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGScale,
    EGShelf,
    EGShelfLayer,
    ObjectType,
    ShelfLayerGeometry,
    SpawnedShelf,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox

# %% names the fitted circuit gives its variables

LAYERS_TEMPLATE_NAME = "layers"
"""
Name of the exchangeable part holding a shelf's layers in a circuit rooted at
:class:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf`.
"""


class LayerVariable(StrEnum):
    """
    Variables a grounded layer circuit carries for the layer itself.

    A nested template's own variables carry no owning-class prefix, unlike the
    ``EGShelfLayer.objects[i].…`` its object slots get when they are grounded.
    """

    THEME = "theme_dominant_type"
    HEIGHT_ABOVE_SHELF_BASE = "height_above_shelf_base"
    RELATIVE_HEIGHT = "relative_height"


_OBJECT_SLOT_PREFIX = f"{EGShelfLayer.__name__}.objects["
"""
Name prefix a grounded layer circuit gives every variable of an object slot, before the
slot's own index.
"""


class ObjectSlotVariable(StrEnum):
    """
    Variables of one object slot that a placement leaves for the model to answer.
    """

    POSITION_X = "position.x"
    POSITION_Y = "position.y"
    YAW = "orientation.z"


# %% the answer


class PlacementRefusal(StrEnum):
    """
    Why a layer cannot take an object.
    """

    TOO_LITTLE_HEADROOM = "the object is taller than the room above the slab"
    FOOTPRINT_TOO_LARGE = "the object does not fit the layer's footprint at any yaw"
    NO_SPACE_LEFT = "what already stands there leaves no room"
    UNSUPPORTED = "the model gives a layer like this holding it no probability"


@dataclass(frozen=True)
class LayerRefusal:
    """
    One layer's reason for not taking an object.
    """

    layer_index: int
    """
    Index of the layer, in :attr:`EGShelf.layers` order.
    """

    reason: PlacementRefusal
    """
    What ruled the layer out.
    """

    room_above_slab: float
    """
    Height available above the layer's slab, in metres.
    """


@dataclass(frozen=True)
class ShelfPlacement:
    """
    Where an object is most likely to belong on a spawned shelf.
    """

    layer_index: int
    """
    Index of the layer the object belongs on, in :attr:`EGShelf.layers` order.
    """

    placed_object: EGObject2D
    """
    The object carrying the position and orientation it was placed at.
    """

    pose: HomogeneousTransformationMatrix
    """
    The placed object's pose in the shelf corpus's frame, at the height its origin takes
    resting on the layer's slab.
    """

    log_likelihood: float
    """
    Log density of this layer holding this object at this pose, which is what the layers
    were compared on.
    """


# %% asking the model


def most_likely_shelf_placement(
    spawned_shelf: SpawnedShelf,
    shelf_circuit: RelationalProbabilisticCircuit,
    held_object: EGObject2D,
    sample_count: int = 1000,
) -> ShelfPlacement:
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
    shelf = spawned_shelf.shelf
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=shelf_circuit.exchangeable_distribution_templates[
            LAYERS_TEMPLATE_NAME
        ].template_distribution
    )
    outcomes = [
        _layer_placement(
            spawned_shelf, registry, layer_index, geometry, held_object, sample_count
        )
        for layer_index, geometry in enumerate(shelf.layer_geometries())
    ]
    placements = [
        outcome for outcome in outcomes if isinstance(outcome, ShelfPlacement)
    ]
    if not placements:
        raise NoShelfPlacementError(
            object_type=held_object.object_type.value,
            object_height=held_object.scale.height,
            refusals=outcomes,
        )
    return max(placements, key=lambda placement: placement.log_likelihood)


def _layer_placement(
    spawned_shelf: SpawnedShelf,
    registry: RelationalCircuitRegistry,
    layer_index: int,
    geometry: ShelfLayerGeometry,
    held_object: EGObject2D,
    sample_count: int,
) -> ShelfPlacement | LayerRefusal:
    """
    Ask the model for the densest pose *held_object* can take on one layer.

    :param spawned_shelf: The shelf standing in a world, whose bodies say what the layer
        already holds.
    :param registry: Registry grounding the fitted layer circuit for a query.
    :param layer_index: Index of the layer to ask about.
    :param geometry: Where that layer's slab sits and how large an object it takes.
    :param held_object: The object to place.
    :param sample_count: Poses drawn to search for the densest one.
    :return: The layer's best placement, or why it cannot take the object at all.
    """
    refused = partial(
        LayerRefusal,
        layer_index=layer_index,
        room_above_slab=geometry.maximum_object_extents.height,
    )
    if geometry.maximum_object_extents.height < held_object.scale.height:
        return refused(reason=PlacementRefusal.TOO_LITTLE_HEADROOM)
    shelf = spawned_shelf.shelf
    occupied = _occupied_footprints(spawned_shelf, layer_index)

    parameters = UnderspecifiedParameters(
        _placement_query(shelf.theme_dominant_type, held_object, len(occupied))
    )
    model = registry.get_model(parameters)
    variables = {variable.name: variable for variable in model.variables}
    slot_prefix = f"{_OBJECT_SLOT_PREFIX}{len(occupied)}]."

    without_neighbours, _ = model.log_conditional(
        _neighbour_evidence(parameters, slot_prefix)
    )
    if without_neighbours is None:
        return refused(reason=PlacementRefusal.UNSUPPORTED)
    conditioned, log_evidence = without_neighbours.log_conditional(
        _layer_and_object_evidence(
            parameters, variables, geometry, shelf.theme_dominant_type, slot_prefix
        )
    )
    if conditioned is None:
        return refused(reason=PlacementRefusal.UNSUPPORTED)

    position_x, position_y, yaw = (
        variables[slot_prefix + variable] for variable in ObjectSlotVariable
    )
    pose_model = conditioned.marginal([position_x, position_y, yaw])
    free_poses = _free_positions(
        shelf, occupied, held_object.scale, position_x, position_y, yaw
    )
    if free_poses.is_empty():
        return refused(
            reason=(
                PlacementRefusal.NO_SPACE_LEFT
                if occupied
                else PlacementRefusal.FOOTPRINT_TOO_LARGE
            )
        )
    free_poses.fill_missing_variables(pose_model.variables)
    truncated, _ = pose_model.log_truncated(free_poses)
    if truncated is None:
        return refused(reason=PlacementRefusal.UNSUPPORTED)

    pose = _densest_pose(truncated, pose_model, sample_count)
    placed_object = dataclasses.replace(
        held_object,
        position=EGPoint2D(x=pose[position_x], y=pose[position_y]),
        orientation=EGRotation(x=0.0, y=0.0, z=pose[yaw]),
        theme_dominant_type=shelf.theme_dominant_type,
    )
    return ShelfPlacement(
        layer_index=layer_index,
        placed_object=placed_object,
        pose=shelf.object_local_pose(
            placed_object, geometry.slab_top_height, spawned_shelf.corpus
        ),
        # Scored against the untruncated layer, so the density is over the same space
        # for every layer and the layers stay comparable.
        log_likelihood=log_evidence + _log_density_at(pose_model, pose),
    )


def _densest_pose(
    free_region_model: ProbabilisticCircuit,
    pose_model: ProbabilisticCircuit,
    sample_count: int,
) -> dict[Variable, float]:
    """
    The pose of highest density within the free region.

    Read straight off the circuit wherever that is possible: a deterministic circuit
    reports its mode exactly, as the box of poses sharing the highest density, and the
    centre of that box is the pose furthest from where the density drops off. Only a
    deterministic circuit can, and truncating to the free region does not always leave
    one, so elsewhere the densest of *sample_count* draws stands in -- close, but a
    slightly different pose from one run to the next.

    :param free_region_model: The pose distribution restricted to the free region.
    :param pose_model: The same distribution untruncated, which ranks the draws.
    :param sample_count: Poses to draw when the mode cannot be read off.
    :return: The chosen value of each pose variable.
    """
    if free_region_model.is_deterministic():
        mode, _ = free_region_model.log_mode()
        widest = max(mode.simple_sets, key=_box_extent)
        return {variable: _middle_of(widest[variable]) for variable in widest.keys()}
    samples = free_region_model.sample(sample_count)
    densest = samples[pose_model.log_likelihood(samples).argmax()]
    return {
        variable: float(densest[index])
        for index, variable in enumerate(pose_model.variables)
    }


def _box_extent(box: SimpleEvent) -> tuple[float, tuple[float, ...]]:
    """
    How much room a box of equally likely poses offers, and where it sits.

    Several boxes can share the highest density, so they are ranked by the room they
    give first -- the widest leaves the most slack for a placement that has to be
    reached for -- and by position second, which only decides ties and does so the same
    way every run.

    :param box: One box of the mode.
    :return: Its extent, and the middle of each of its variables.
    """
    middles = tuple(_middle_of(box[variable]) for variable in box.keys())
    extent = 1.0
    for variable in box.keys():
        widest = max(
            box[variable].simple_sets, key=lambda piece: piece.upper - piece.lower
        )
        extent *= widest.upper - widest.lower
    return extent, middles


def _middle_of(value: Interval) -> float:
    """
    The middle of the widest stretch a variable takes in a box.

    A box assigns each variable a set that may come in several stretches; any value from
    any of them is part of the box, so the widest one is taken, being the one whose
    middle sits furthest from an edge.

    :param value: What the box assigns one variable.
    :return: The middle of its widest stretch.
    """
    widest = max(value.simple_sets, key=lambda piece: piece.upper - piece.lower)
    return (widest.lower + widest.upper) / 2


def _log_density_at(
    pose_model: ProbabilisticCircuit, pose: dict[Variable, float]
) -> float:
    """
    The log density *pose_model* gives one pose.

    :param pose_model: The untruncated pose distribution.
    :param pose: A value for each of its variables.
    :return: The log density there.
    """
    point = np.array([[pose[variable] for variable in pose_model.variables]])
    return float(pose_model.log_likelihood(point)[0])


def _placement_query(
    theme_dominant_type: ObjectType, held_object: EGObject2D, occupied_count: int
):
    """
    Build an EGShelfLayer query for a layer holding *occupied_count* objects plus the
    held one.

    The objects already standing there get plain free slots rather than their measured
    poses: objects on a layer are conditionally independent given the layer, so pinning
    them cannot move the held object away from them -- it would only add their own
    typicality to the score, which says nothing about where the held object belongs.
    Their count is what does reach the held object, through the layer's aggregation
    statistics, so the slots are still there.

    :param theme_dominant_type: The shelf's dominant object type.
    :param held_object: The object whose type and size are pinned, and whose position
        and yaw are left for the model to answer.
    :param occupied_count: How many objects the layer already holds.
    :return: An underspecified EGShelfLayer query whose last object slot is the held
        one.
    """
    held_slot = a(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=held_object.object_type,
        scale=held_object.scale,
        position=a(EGPoint2D)(x=..., y=...),
        # Roll and pitch are pinned upright, as everywhere else the model is asked;
        # only yaw genuinely varies.
        orientation=a(EGRotation)(x=0.0, y=0.0, z=...),
        source_id=None,
        theme_dominant_type=theme_dominant_type,
    )
    return a(EGShelfLayer)(
        objects=[free_object_slot(theme_dominant_type) for _ in range(occupied_count)]
        + [held_slot],
        theme_dominant_type=theme_dominant_type,
        height_above_shelf_base=...,
        relative_height=...,
        vertical_clearance=...,
    )


def _neighbour_evidence(
    parameters: UnderspecifiedParameters, held_slot_prefix: str
) -> dict[Variable, Any]:
    """
    What the query pins about the objects already on the layer.

    Conditioned on and then discarded, so their fixed upright orientation and theme do
    not count towards the layer's score: each of them costs the same fixed amount of log
    density, which would make a layer look worse purely for holding more.

    :param parameters: Parameters of the placement query.
    :param held_slot_prefix: Variable-name prefix of the held object's own slot.
    :return: Assignments belonging to the other object slots.
    """
    return {
        variable: value
        for variable, value in parameters.conditioning_assignments_from_literal_values.items()
        if variable.name.startswith(_OBJECT_SLOT_PREFIX)
        and not variable.name.startswith(held_slot_prefix)
    }


def _layer_and_object_evidence(
    parameters: UnderspecifiedParameters,
    variables: dict[str, Variable],
    geometry: ShelfLayerGeometry,
    theme_dominant_type: ObjectType,
    held_slot_prefix: str,
) -> dict[Variable, Any]:
    """
    What is known about the layer itself and about the held object, as circuit
    assignments.

    This is what the layers are scored on, so it holds nothing about the objects already
    standing there; those are :func:`_neighbour_evidence`.

    The layer's own fields are keyed by the circuit's variables directly rather than
    taken from *parameters*: :class:`RelationalCircuitRegistry` qualifies a query's
    variable names with its circuit's ``class_``, which for a nested template is the DAO
    class, so the assignments it derives for them match no variable and are dropped
    without a word. The object slots are unaffected, since grounding already gave those
    the names the query uses.

    The layer's ``vertical_clearance`` is left out: a layer resting on the shelf's top
    has open air above it, and no value stands for that.

    :param parameters: Parameters of the placement query, holding the object slots' own
        assignments.
    :param variables: The grounded circuit's variables, by name.
    :param geometry: Where the layer's slab sits.
    :param theme_dominant_type: The shelf's dominant object type.
    :param held_slot_prefix: Variable-name prefix of the held object's own slot.
    :return: Assignments to condition the grounded circuit on.
    """
    held_object_assignments = {
        variable: value
        for variable, value in parameters.conditioning_assignments_from_literal_values.items()
        if variable.name.startswith(held_slot_prefix)
    }
    return {
        **held_object_assignments,
        variables[LayerVariable.THEME]: theme_dominant_type,
        variables[
            LayerVariable.HEIGHT_ABOVE_SHELF_BASE
        ]: geometry.height_above_shelf_base,
        variables[LayerVariable.RELATIVE_HEIGHT]: geometry.relative_height,
    }


# %% the space a layer has left


def _occupied_footprints(
    spawned_shelf: SpawnedShelf, layer_index: int
) -> list[BoundingBox]:
    """
    Footprints of the bodies standing on one layer, in the shelf corpus's frame.

    Read from the world rather than from the layer's drawn objects, so an object the
    layout resolver dropped or stood in for takes exactly the space it really takes.

    :param spawned_shelf: The shelf standing in a world.
    :param layer_index: Index of the layer whose bodies are measured.
    :return: One bounding box per body on the layer.
    """
    return [
        body.collision.as_bounding_box_collection_in_frame(
            spawned_shelf.corpus
        ).bounding_box()
        for body in spawned_shelf.layers[layer_index].object_bodies.values()
    ]


YAW_BAND_DEGREES = 15
"""
Width of the yaw bands the free region is built from.

How much of a layer an object covers depends on how it is turned, so the region a
placement is drawn from pairs each band of yaw with the positions that fit at *any* yaw
inside it. Narrower bands describe the true region more closely, at one more piece of
the region each; 15 degrees keeps an elongated object from being rejected by the corner
it would only reach when turned diagonally, which is what a single yaw-free bound does.
"""


def _widest_footprint_in_band(
    scale: EGScale, from_degrees: float, to_degrees: float
) -> tuple[float, float]:
    """
    The most of a layer's depth and face an object of *scale* can cover while turned to
    any yaw between *from_degrees* and *to_degrees*.

    An object turned by yaw covers ``length·|cos| + width·|sin|`` of the depth, which
    peaks either at an end of the band or where the two terms balance, so those are the
    angles worth measuring.

    :param scale: Size of the object.
    :param from_degrees: Start of the yaw band.
    :param to_degrees: End of the yaw band.
    :return: The widest extent along the layer's depth and along its face.
    """
    balance_point = math.degrees(math.atan2(scale.width, scale.length))
    critical = [from_degrees, to_degrees] + [
        balance_point + quarter_turn * 90
        for quarter_turn in range(-4, 5)
        if from_degrees <= balance_point + quarter_turn * 90 <= to_degrees
    ]
    radians = [math.radians(degrees) for degrees in critical]
    return (
        max(
            scale.length * abs(math.cos(angle)) + scale.width * abs(math.sin(angle))
            for angle in radians
        ),
        max(
            scale.length * abs(math.sin(angle)) + scale.width * abs(math.cos(angle))
            for angle in radians
        ),
    )


def _free_positions(
    shelf: EGShelf,
    occupied: list[BoundingBox],
    scale: EGScale,
    position_x: Variable,
    position_y: Variable,
    yaw: Variable,
) -> Event:
    """
    The poses on a layer at which an object of *scale* stands inside the layer and clear
    of everything already on it.

    The layer's own bounds are paired with the yaw they hold for, so an object longer
    than the layer is deep can still be placed along it -- a single bound covering
    every yaw would have to allow for the diagonal and would turn such an object away
    from a layer it plainly fits. Clearance from what already stands there does use
    that yaw-free radius: being generous costs a little space, while being wrong puts
    the object inside its neighbour.

    :param shelf: The shelf whose footprint bounds the layer.
    :param occupied: Footprints already taken, in the corpus frame.
    :param scale: Size of the object to place.
    :param position_x: The circuit's variable for the object's x position.
    :param position_y: The circuit's variable for the object's y position.
    :param yaw: The circuit's variable for the object's yaw, in degrees.
    :return: The free poses, empty when the layer has none left -- including when the
        object does not fit the layer's footprint at any yaw.
    """
    # The content frame's x-axis spans the shelf's length and y spans its width, the
    # same mapping object_local_pose uses -- not shelf.scale's own width/length order.
    bands = []
    for band_start in range(-180, 180, YAW_BAND_DEGREES):
        band_end = band_start + YAW_BAND_DEGREES
        depth_covered, face_covered = _widest_footprint_in_band(
            scale, band_start, band_end
        )
        half_x = shelf.scale.length / 2 - depth_covered / 2
        half_y = shelf.scale.width / 2 - face_covered / 2
        if half_x <= 0 or half_y <= 0:
            continue
        bands.append(
            SimpleEvent.from_data(
                {
                    yaw: closed(band_start, band_end),
                    position_x: closed(-half_x, half_x),
                    position_y: closed(-half_y, half_y),
                }
            )
        )
    if not bands:
        return Event.from_simple_sets()

    free = Event.from_simple_sets(*bands)
    footprint_radius = math.hypot(scale.length, scale.width) / 2
    for box in occupied:
        # Read through the intervals rather than off min_x/max_x, which are stated
        # relative to the box's own origin.
        taken_x, taken_y = box.x_interval, box.y_interval
        taken = SimpleEvent.from_data(
            {
                position_x: closed(
                    taken_x.lower - footprint_radius, taken_x.upper + footprint_radius
                ),
                position_y: closed(
                    taken_y.lower - footprint_radius, taken_y.upper + footprint_radius
                ),
            }
        ).as_composite_set()
        taken.fill_missing_variables([yaw])
        free -= taken
    return free
