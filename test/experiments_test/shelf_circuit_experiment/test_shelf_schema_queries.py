from __future__ import annotations

import dataclasses

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import PlanarBoundingBox, Scale
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
)

from .shelf_dataset import random_shelves


# %% helpers
def _book(x: float, y: float) -> RelationalCircuitExperimentObject2D:
    return RelationalCircuitExperimentObject2D(
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.15, y=0.12, z=0.18),
        pose=Pose2D(x=x, y=y, yaw=0.4),
        source_id="book",
    )


def _training_layer(
    objects: list[RelationalCircuitExperimentObject2D],
) -> RelationalCircuitExperimentShelfLayer:
    """
    :param objects: The objects to stand on the layer.
    :return: A layer whose own attributes are those of a layer the shelf model was
        fitted on, so the model supports them, holding *objects*.
    """
    return dataclasses.replace(random_shelves()[0].layers[0], objects=objects)


def _free_space(world: World, minimum_x: float, maximum_x: float):
    """
    :return: A planar free space spanning x in [*minimum_x*, *maximum_x*] and y in
        [-0.3, 0.3] in the frame of the world root.
    """
    free_space = PlanarGraphOfBoundingBoxes(world=world)
    free_space.add_node(
        PlanarBoundingBox(
            min_x=minimum_x,
            min_y=-0.3,
            max_x=maximum_x,
            max_y=0.3,
            origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        )
    )
    return free_space


# %% shelf query
def test_an_underspecified_shelf_has_the_requested_layers_and_theme(
    shelf_model,
) -> None:
    shelf = shelf_model.shelf_backend().generate_one(
        RelationalCircuitExperimentShelf.underspecified_query(ObjectType.BOTTLE, [2, 1])
    )

    assert [len(layer.objects) for layer in shelf.layers] == [2, 1]
    assert shelf.theme_dominant_type is ObjectType.BOTTLE
    assert {layer.theme_dominant_type for layer in shelf.layers} == {ObjectType.BOTTLE}


def test_every_object_of_an_underspecified_shelf_gets_a_pose(shelf_model) -> None:
    shelf = shelf_model.shelf_backend().generate_one(
        RelationalCircuitExperimentShelf.underspecified_query(ObjectType.BOOK, [2])
    )

    assert all(
        isinstance(object_.pose, Pose2D)
        for layer in shelf.layers
        for object_ in layer.objects
    )


# %% layer placement query
def test_a_placed_object_keeps_its_type_and_scale_and_lands_in_the_free_space(
    shelf_model,
) -> None:
    world = World.create_with_root_body()
    layer = _training_layer([_book(-0.2, 0.0)])
    held_object = _book(0.0, 0.0)

    placed_layer = shelf_model.layer_backend().generate_one(
        layer.placement_query(
            layer.objects, held_object.placement_query(), _free_space(world, 0.1, 0.3)
        )
    )

    placed_object = placed_layer.objects[-1]
    assert placed_object.object_type is held_object.object_type
    assert placed_object.scale == held_object.scale
    assert 0.1 <= float(placed_object.pose.x) <= 0.3


def test_a_placement_query_pins_the_yaw_it_is_given(shelf_model) -> None:
    world = World.create_with_root_body()
    layer = _training_layer([])

    placed_layer = shelf_model.layer_backend().generate_one(
        layer.placement_query(
            [], _book(0.0, 0.0).placement_query(yaw=0.0), _free_space(world, -0.3, 0.3)
        )
    )

    assert float(placed_layer.objects[-1].pose.yaw) == 0.0


# %% evidence
def test_an_evidence_query_holds_the_scale_and_pose_of_its_object(shelf_model) -> None:
    world = World.create_with_root_body()
    standing_book = _book(-0.25, 0.1)
    layer = _training_layer([standing_book])

    placed_layer = shelf_model.layer_backend().generate_one(
        layer.placement_query(
            [standing_book],
            _book(0.0, 0.0).placement_query(),
            _free_space(world, 0.0, 0.3),
        )
    )

    evidence_object = placed_layer.objects[0]
    assert evidence_object.scale == standing_book.scale
    assert [
        float(evidence_object.pose.x),
        float(evidence_object.pose.y),
        float(evidence_object.pose.yaw),
    ] == [
        float(standing_book.pose.x),
        float(standing_book.pose.y),
        float(standing_book.pose.yaw),
    ]
