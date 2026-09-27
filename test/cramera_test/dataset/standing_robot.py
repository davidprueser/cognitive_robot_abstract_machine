"""
A robot whose root sits above its lowest point, the way a humanoid's pelvis rides above
its feet, built in code so no description file is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Self

from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

STANDING_ROBOT_HEIGHT = 0.4
"""
How tall the robot's one box is; its root sits at the box's middle, so its lowest point
is half of this below the root.
"""


@dataclass(eq=False)
class StandingRobot(MinimalRobot):
    """
    A robot of one box centred on its root.
    """

    @classmethod
    def from_description(cls, prefix: str | None = None) -> Self:
        world = World.create_with_root_body("standing_robot_root", prefix)
        box = Box(
            origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            scale=Scale(0.2, 0.2, STANDING_ROBOT_HEIGHT),
        )
        world.root.collision = ShapeCollection([box], reference_frame=world.root)
        return cls.from_world(world)
