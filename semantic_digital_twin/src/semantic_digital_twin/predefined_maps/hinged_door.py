from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Handle,
)
from semantic_digital_twin.specifications.connections import (
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class HingeSide(IntEnum):
    """
    The side edge of a door leaf its hinge is on, valued as the sign of the leaf's
    y-axis at that edge.
    """

    NEGATIVE_Y = -1
    POSITIVE_Y = 1


@dataclass
class HingedDoor:
    """
    World-independent description of a door leaf swinging about a vertical hinge on one
    of its side edges, with a handle near the other one.

    The door's origin is the centre of its leaf. The handle sits on the leaf's negative
    x face, and opening swings the leaf towards that face.
    """

    scale: Scale = field(default_factory=lambda: Door.default_scale)
    """
    Extents of the leaf: thickness, width and height.
    """

    hinge_side: HingeSide = HingeSide.NEGATIVE_Y
    """
    The side edge the hinge is on; the handle sits near the opposite one.
    """

    handle_height: float = 1.0
    """
    Height of the handle's centre above the leaf's bottom edge.
    """

    handle_distance_to_edge: float = 0.1
    """
    Distance of the handle's centre from the leaf's side edge it sits near.
    """

    opening_angle: float = np.pi / 2
    """
    Angle the leaf swings by from closed to fully open, in radians.
    """

    color: Color = field(default_factory=Color.RED)
    """
    Color of the leaf, setting it apart from the wall around it.
    """

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Door:
        """
        Materialize this door in ``world``, closed and hung on its hinge.

        :param world: The world the door is added to.
        :param name: Prefix for the names of every entity this door creates.
        :param parent: The entity the hinge attaches to. If None, ``world.root`` is
            used.
        :param parent_T_self: Placement of the leaf's centre in ``parent``'s frame while
            the door is closed. Identity if None.
        :return: The door annotation.
        """
        handle_root_specification = (
            Handle.get_default_root_kinematic_structure_entity_specification()
        )
        handle_root_specification.parent_T_self = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                -self.scale.x / 2,
                -self.hinge_side * (self.scale.y / 2 - self.handle_distance_to_edge),
                self.handle_height - self.scale.z / 2,
            )
        )
        door = Door.get_annotation_specification(
            name,
            Door.get_default_root_kinematic_structure_entity_specification(
                scale=self.scale
            ),
            part_specifications={
                "handle": Handle.get_annotation_specification(
                    f"{name}_handle", handle_root_specification
                )
            },
        ).spawn(world, parent=parent, parent_T_self=parent_T_self)
        door.root.dye(self.color)
        door.mount_on_movable_joint(
            RevoluteConnectionSpecification(
                axis=Vector3.Z() * -self.hinge_side,
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=0.0, upper_position=self.opening_angle
                ),
            )
        )
        return door
