from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.elevator_room import (
    LabElevatorRoom,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

# %% the lab as a building with an elevator


@dataclass
class LabBuilding(ElevatorBuilding):
    """
    The foyer of the ECOTEC V building with its elevator, the landing in front of the
    elevator on the first floor and the hallway, placed in the map frame of the lab
    occupancy grid map.
    """

    elevator_room: LabElevatorRoom = field(default_factory=LabElevatorRoom)
    """
    The foyer the elevator stands in, together with the landing and the hallway.
    """

    robot_start_distance_from_doors: float = 2.0
    """
    How far in front of the elevator's doors a simulated robot starts.
    """

    @property
    def world_T_robot_start(self) -> HomogeneousTransformationMatrix:
        """
        :attr:`robot_start_distance_from_doors` in front of the middle of the elevator's
        doors, on the ground floor, facing the cabin.
        """
        cabin_scale = self.elevator_room.elevator.scale
        elevator_T_robot_start = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-cabin_scale.x / 2 - self.robot_start_distance_from_doors,
            z=-cabin_scale.z / 2,
        )
        return (
            self.elevator_room.map_T_room
            @ self.elevator_room.room_T_elevator
            @ elevator_T_robot_start
        )

    def spawn(self, world: World, name: str = "lab") -> Elevator:
        """
        :meth:`ElevatorBuilding.spawn`, with the names of every entity this building
        creates prefixed by ``name``.
        """
        room = self.elevator_room.spawn(world, name)
        return next(
            elevator
            for elevator in world.get_semantic_annotations_by_type(Elevator)
            if elevator.root.parent_kinematic_structure_entity is room.floor.root
        )
