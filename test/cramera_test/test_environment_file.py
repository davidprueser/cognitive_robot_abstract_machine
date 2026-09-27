"""
An environment is read by the parser its file asks for, and the robots put into it stand
on its floor.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cramera.environment_file import (
    EnvironmentFile,
    GazeboEnvironmentFile,
    UnsupportedEnvironmentFileError,
    URDFEnvironmentFile,
    USDSceneEnvironmentFile,
)
from cramera.multi_robot import RobotInstance, RobotScene, move_robot_to
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

from .dataset.standing_robot import StandingRobot

DATASET = Path(__file__).parent / "dataset"
FLOOR_SLAB = DATASET / "floor_slab.urdf"


def floor_slab_top() -> float:
    """
    :return: How high the top of the slab ``floor_slab.urdf`` describes is.
    """
    slab = URDFParser.from_file(str(FLOOR_SLAB)).parse()
    return max(
        body.collision.as_bounding_box_collection_in_frame(slab.root)
        .bounding_box()
        .max_z
        for body in slab.bodies
        if body.collision
    )


# %% which parser a file is read by


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("kitchen.urdf", URDFEnvironmentFile),
        ("small_warehouse.world", GazeboEnvironmentFile),
        ("model.sdf", GazeboEnvironmentFile),
        ("world.usda", USDSceneEnvironmentFile),
        ("world.usd", USDSceneEnvironmentFile),
        ("world.usdc", USDSceneEnvironmentFile),
        ("scan.usdz", USDSceneEnvironmentFile),
    ],
)
def test_an_environment_file_is_read_as_its_suffix_says(
    file_name: str, expected: type[EnvironmentFile]
) -> None:
    assert type(EnvironmentFile.from_path(file_name)) is expected


def test_a_package_url_is_read_as_its_suffix_says() -> None:
    environment = EnvironmentFile.from_path(
        "package://aws_robomaker_small_warehouse_world/worlds/small.world"
    )

    assert type(environment) is GazeboEnvironmentFile


def test_a_file_no_parser_reads_is_refused() -> None:
    with pytest.raises(UnsupportedEnvironmentFileError):
        EnvironmentFile.from_path("notes.txt")


# %% standing on the floor


def lowest_point_of(world: World, identifier: str) -> float:
    [robot] = [
        robot
        for robot in world.get_semantic_annotations_by_type(StandingRobot)
        if robot.root.name.prefix == identifier
    ]
    return world.height_of_lowest_collision_point_of_branch(robot.root)


def standing_scene(x: float) -> RobotScene:
    return RobotScene(
        instances=[
            RobotInstance(
                identifier="standing",
                label="Standing robot",
                robot_type=StandingRobot,
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=x),
            )
        ],
        active_identifier="standing",
    )


def test_a_robot_above_a_floor_stands_on_its_top() -> None:
    world = standing_scene(x=0.0).build_world(str(FLOOR_SLAB))

    assert lowest_point_of(world, "standing") == pytest.approx(
        floor_slab_top(), abs=1e-6
    )


def test_a_robot_beside_every_floor_stands_on_the_ground() -> None:
    world = standing_scene(x=10.0).build_world(str(FLOOR_SLAB))

    assert lowest_point_of(world, "standing") == pytest.approx(0.0, abs=1e-6)


def test_a_robot_in_no_environment_stands_on_the_ground() -> None:
    world = standing_scene(x=0.0).build_world()

    assert lowest_point_of(world, "standing") == pytest.approx(0.0, abs=1e-6)


# %% moving a robot elsewhere


def test_a_moved_robot_stands_where_it_was_moved_to() -> None:
    world = standing_scene(x=10.0).build_world(str(FLOOR_SLAB))
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)

    move_robot_to(world, robot, x=0.5, y=-0.5, yaw=1.0)

    world_T_root = robot.root.global_pose.to_np()
    assert world_T_root[:2, 3] == pytest.approx([0.5, -0.5])
    assert math.atan2(world_T_root[1, 0], world_T_root[0, 0]) == pytest.approx(1.0)


def test_a_robot_moved_onto_a_floor_stands_on_it() -> None:
    world = standing_scene(x=10.0).build_world(str(FLOOR_SLAB))
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)

    move_robot_to(world, robot, x=0.0, y=0.0, yaw=0.0)

    assert lowest_point_of(world, "standing") == pytest.approx(
        floor_slab_top(), abs=1e-6
    )


def test_a_robot_moved_off_a_floor_stands_on_the_ground() -> None:
    world = standing_scene(x=0.0).build_world(str(FLOOR_SLAB))
    [robot] = world.get_semantic_annotations_by_type(StandingRobot)

    move_robot_to(world, robot, x=10.0, y=0.0, yaw=0.0)

    assert lowest_point_of(world, "standing") == pytest.approx(0.0, abs=1e-6)


# %% the environment's own joints


def test_a_scene_stands_the_environments_joints_where_it_is_told() -> None:
    door = str(DATASET / "hinged_door.urdf")
    [hinge] = [
        connection
        for connection in standing_scene(x=0.0).build_world(door).connections
        if connection.child.name.name == "leaf"
    ]
    scene = standing_scene(x=0.0)
    scene.environment_joint_positions = {str(hinge.name): 1.2}

    world = scene.build_world(door)

    [opened] = [c for c in world.connections if str(c.name) == str(hinge.name)]
    assert opened.position == pytest.approx(1.2)
