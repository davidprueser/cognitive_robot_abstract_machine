from __future__ import annotations

from dataclasses import dataclass

import pytest

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
import experiments.scene_generation_experiments.demo as demo_module
from coraplex.datastructures.dataclasses import Context
from experiments.scene_generation_experiments.demo import move_to_reach_shelf
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


@dataclass
class _RobotWithEndEffector:
    """
    Stands in for the parts of :class:`AbstractRobot` :func:`move_to_reach_shelf` reads --
    just enough to build the ``GraspDescription`` it hands to ``MoveToReach``, without the
    real ``HSRB`` semantic annotation this module's assertions never look at.
    """

    end_effector: EndEffector


@pytest.fixture
def spawned_shelf() -> EGShelf:
    """
    A two-layer shelf, standing at the root of a fresh world.
    """
    shelf = EGShelf(
        scale=Scale(x=0.4, y=0.8, z=1.2),
        layers=[
            EGShelfLayer(objects=[], theme_dominant_type=ObjectType.BOOK)
            for _ in range(2)
        ],
        theme_dominant_type=ObjectType.BOOK,
    )
    world = World.create_with_root_body("map")
    shelf.spawn(world=world, parent=world.root)
    return shelf


def test_reach_target_height_matches_the_layers_slab_top(
    spawned_shelf: EGShelf,
) -> None:
    """
    The end-effector target has to land on the slab's own top surface -- the height
    :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.object_local_pose`
    seats objects at -- rather than the corpus-center-relative height silently produced by
    passing a shelf-base-relative height straight into a corpus-frame pose.
    """
    demo_module.context = Context(
        world=spawned_shelf.world,
        robot=_RobotWithEndEffector(end_effector=None),
    )
    layer = spawned_shelf.layers[0]
    expected_slab_top_height = spawned_shelf.layer_geometries()[0].slab_top_height
    placed_object = EGObject2D(
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.04, y=0.12, z=0.2),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id=None,
    )

    action = move_to_reach_shelf(
        spawned_shelf, placed_object, str(layer.annotation.root.name)
    )

    assert float(
        action.target_pose_end_effector.to_position().z
    ) == pytest.approx(expected_slab_top_height)
