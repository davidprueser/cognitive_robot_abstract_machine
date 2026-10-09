from __future__ import annotations

import dataclasses

import pytest

from experiments.shelf_generation_experiments.placement.exceptions import (
    NoShelfPlacementError,
)
from experiments.shelf_generation_experiments.placement.shelf_placement import (
    ShelfPlacement,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import random_shelves


# %% fixtures
@pytest.fixture
def empty_spawned_shelf() -> RelationalCircuitExperimentShelf:
    """
    A shelf of the training data, emptied and spawned into a fresh world.
    """
    training_shelf = random_shelves()[0]
    shelf = dataclasses.replace(
        training_shelf,
        layers=[
            dataclasses.replace(layer, objects=[]) for layer in training_shelf.layers
        ],
        source_ids=[],
    )
    shelf.spawn(World.create_with_root_body())
    return shelf


def _held_book(height: float) -> RelationalCircuitExperimentObject2D:
    return RelationalCircuitExperimentObject2D(
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.15, y=0.12, z=height),
        pose=Pose2D(),
        source_id="book",
    )


# %% placement
def test_a_placement_keeps_the_held_objects_type_and_size(
    shelf_model, empty_spawned_shelf
) -> None:
    held_object = _held_book(0.15)

    placement = ShelfPlacement(
        shelf=empty_spawned_shelf, model=shelf_model
    ).most_likely_placement(held_object)

    assert placement.placed_object.object_type is held_object.object_type
    assert placement.placed_object.scale == held_object.scale
    assert any(placement.layer is layer for layer in empty_spawned_shelf.layers)


def test_a_placement_keeps_the_held_object_inside_its_layer(
    shelf_model, empty_spawned_shelf
) -> None:
    held_object = _held_book(0.15)

    placement = ShelfPlacement(
        shelf=empty_spawned_shelf, model=shelf_model
    ).most_likely_placement(held_object, yaw=0.0)

    half_x = empty_spawned_shelf.scale.x / 2
    half_y = empty_spawned_shelf.scale.y / 2
    assert abs(float(placement.placed_object.pose.x)) <= half_x
    assert abs(float(placement.placed_object.pose.y)) <= half_y
    assert float(placement.placed_object.pose.yaw) == 0.0


def test_an_object_taller_than_every_layer_has_no_placement(
    shelf_model, empty_spawned_shelf
) -> None:
    held_object = _held_book(height=empty_spawned_shelf.scale.z)

    with pytest.raises(NoShelfPlacementError) as error:
        ShelfPlacement(
            shelf=empty_spawned_shelf, model=shelf_model
        ).most_likely_placement(held_object)

    assert error.value.object_type is held_object.object_type
