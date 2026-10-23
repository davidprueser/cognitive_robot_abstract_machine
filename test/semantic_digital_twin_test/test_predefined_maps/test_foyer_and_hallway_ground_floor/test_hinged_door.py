import numpy as np
import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.hinged_door import (
    HingedDoor,
    HingeSide,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture(params=list(HingeSide))
def hinged_door(request) -> HingedDoor:
    return HingedDoor(hinge_side=request.param)


@pytest.fixture()
def hinged_door_world(hinged_door) -> tuple[World, Door]:
    world = World.create_with_root_body("root")
    door = hinged_door.spawn(world, "door")
    return world, door


# %% swinging


def test_handle_sits_on_the_side_opposite_the_hinge(hinged_door, hinged_door_world):
    world, door = hinged_door_world
    door_P_handle = world.compute_forward_kinematics_np(door.root, door.handle.root)
    assert np.sign(door_P_handle[1, 3]) == -hinged_door.hinge_side


def test_opening_swings_the_door_about_its_hinge_side_edge(
    hinged_door, hinged_door_world
):
    world, door = hinged_door_world
    door_P_hinge = np.array([0, hinged_door.hinge_side * hinged_door.scale.y / 2, 0, 1])
    closed_hinge = world.compute_forward_kinematics_np(world.root, door.root) @ (
        door_P_hinge
    )
    door.movable_joint.position = door.movable_joint.dof.limits.upper.position
    opened_hinge = world.compute_forward_kinematics_np(world.root, door.root) @ (
        door_P_hinge
    )
    assert opened_hinge == pytest.approx(closed_hinge)


def test_opening_turns_the_door_by_its_opening_angle(hinged_door, hinged_door_world):
    world, door = hinged_door_world
    door.movable_joint.position = door.movable_joint.dof.limits.upper.position
    root_T_door = world.compute_forward_kinematics_np(world.root, door.root)
    assert abs(np.arctan2(root_T_door[1, 0], root_T_door[0, 0])) == pytest.approx(
        hinged_door.opening_angle
    )


def test_opening_swings_the_door_towards_its_handle(hinged_door_world):
    world, door = hinged_door_world
    handle_side = np.sign(
        world.compute_forward_kinematics_np(door.root, door.handle.root)[0, 3]
    )
    door.movable_joint.position = door.movable_joint.dof.limits.upper.position
    opened_center = world.compute_forward_kinematics_np(world.root, door.root)[:3, 3]
    assert np.sign(opened_center[0]) == handle_side


# %% appearance


def test_door_takes_its_color(hinged_door, hinged_door_world):
    _, door = hinged_door_world
    door_colors = {
        shape.color for shape in door.root.collision.shapes + door.root.visual.shapes
    }
    assert door_colors == {hinged_door.color}
