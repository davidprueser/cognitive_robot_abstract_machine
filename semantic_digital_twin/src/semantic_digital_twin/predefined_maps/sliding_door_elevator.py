from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
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

if TYPE_CHECKING:
    from semantic_digital_twin.specifications.semantic_annotations import (
        SemanticAnnotationWithRootSpecification,
    )


class DoorSide(IntEnum):
    """
    The half of an elevator front a door covers, valued as the sign of the cabin's
    y-axis on that half.
    """

    NEGATIVE_Y = -1
    POSITIVE_Y = 1


@dataclass
class SlidingDoorElevator:
    """
    World-independent description of an elevator cabin that travels vertically and is
    entered through two doors sliding apart in its open front face.

    The cabin's origin is its centre, and its front faces the cabin's negative x-axis.
    """

    scale: Scale = field(default_factory=lambda: Scale(2, 2, 2))
    """
    Outer extents of the cabin: depth from the front to the back, width across the
    front, and height.
    """

    wall_thickness: float = 0.05
    """
    Thickness of the cabin walls and of each door.
    """

    maximum_speed: float = 1.0
    """
    Highest speed the cabin travels at, in metres per second.
    """

    cabin_color: Color = field(default_factory=Color.CYAN)
    """
    Color of the cabin.
    """

    door_color: Color = field(default_factory=Color.RED)
    """
    Color of the doors, setting them apart from the cabin and the walls around it.
    """

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Elevator:
        """
        Materialize this elevator in ``world``: a cabin on a vertical drive, carrying
        two doors that each slide open by their own width.

        :param world: The world the elevator is added to.
        :param name: Prefix for the names of every entity this elevator creates.
        :param parent: The entity the drive attaches to. If None, ``world.root`` is
            used.
        :param parent_T_self: Placement of the cabin's centre in ``parent``'s frame
            while the drive is at zero. Identity if None.
        :return: The elevator annotation.
        """
        elevator = Elevator.get_annotation_specification(
            name,
            Elevator.get_default_root_kinematic_structure_entity_specification(
                scale=self.scale, wall_thickness=self.wall_thickness
            ),
            parent_connection_specification=PrismaticConnectionSpecification(
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    maximum_speed=self.maximum_speed
                ),
            ),
            part_specifications={
                "doors": [
                    self._door_specification(f"{name}_door_{index}", side)
                    for index, side in enumerate(DoorSide)
                ]
            },
        ).spawn(world, parent=parent, parent_T_self=parent_T_self)
        elevator.root.dye(self.cabin_color)
        for door in elevator.doors:
            door.root.dye(self.door_color)
        return elevator

    def _door_specification(
        self, name: str, side: DoorSide
    ) -> SemanticAnnotationWithRootSpecification[Door]:
        """
        Describe the door covering one half of the front face.

        :param name: The name of the door.
        :param side: The half of the front the door covers; the door slides further
            towards that side when opening.
        :return: The door's annotation specification.
        """
        door_scale = Scale(self.wall_thickness, self.scale.y / 2, self.scale.z)
        root_specification = (
            Door.get_default_root_kinematic_structure_entity_specification(
                scale=door_scale,
                connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.Y() * side,
                    dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                        lower_position=0.0, upper_position=door_scale.y
                    ),
                ),
            )
        )
        root_specification.parent_T_self = HomogeneousTransformationMatrix.from_xyz_rpy(
            -self.scale.x / 2, side * self.scale.y / 4, 0
        )
        return Door.get_annotation_specification(name, root_specification)
