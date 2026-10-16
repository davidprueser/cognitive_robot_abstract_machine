import pytest

from semantic_digital_twin.predefined_maps.sliding_door_elevator import (
    SlidingDoorElevator,
)
from semantic_digital_twin.world import World


@pytest.fixture()
def sliding_door_elevator() -> SlidingDoorElevator:
    return SlidingDoorElevator()


def test_opening_slides_each_door_away_from_the_middle_by_its_width(
    sliding_door_elevator,
):
    world = World.create_with_root_body("root")
    elevator = sliding_door_elevator.spawn(world, "elevator")

    def door_offsets() -> list[float]:
        return [
            float(world.compute_forward_kinematics_np(elevator.root, door.root)[1, 3])
            for door in elevator.doors
        ]

    closed_offsets = door_offsets()
    elevator.open()
    opened_offsets = door_offsets()

    door_width = sliding_door_elevator.scale.y / 2
    assert [
        opened - closed for opened, closed in zip(opened_offsets, closed_offsets)
    ] == pytest.approx([-door_width, door_width])


def test_cabin_and_doors_take_their_colors(sliding_door_elevator):
    world = World.create_with_root_body("root")
    elevator = sliding_door_elevator.spawn(world, "elevator")
    cabin_colors = {
        shape.color
        for shape in elevator.root.collision.shapes + elevator.root.visual.shapes
    }
    door_colors = {
        shape.color
        for door in elevator.doors
        for shape in door.root.collision.shapes + door.root.visual.shapes
    }
    assert cabin_colors == {sliding_door_elevator.cabin_color}
    assert door_colors == {sliding_door_elevator.door_color}
