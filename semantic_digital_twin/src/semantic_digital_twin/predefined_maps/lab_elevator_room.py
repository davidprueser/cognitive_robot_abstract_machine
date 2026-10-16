from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.predefined_maps.rectilinear_room import (
    Doorway,
    OpenEdge,
    RectilinearRoom,
)
from semantic_digital_twin.predefined_maps.sliding_door_elevator import (
    SlidingDoorElevator,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


def lab_elevator_room_outline() -> list[Point2]:
    """
    :return: The inner wall faces of the room, traced from the lab occupancy grid map
        counter-clockwise from the inner corner where the left and bottom walls meet.
        The pocket between the vertices at x 2.25 and 3.39 is the elevator shaft, the
        recess in the left wall between y 5.325 and 6.74 holds a doorway.
    """
    return [
        Point2(0.000, 0.000),
        Point2(0.985, 0.000),
        Point2(0.985, -0.145),
        Point2(3.045, -0.145),
        Point2(3.045, -0.005),
        Point2(5.255, -0.005),
        Point2(5.255, 0.065),
        Point2(7.055, 0.065),
        Point2(7.055, -0.065),
        Point2(9.500, -0.065),
        Point2(9.500, 12.070),
        Point2(6.550, 12.070),
        Point2(6.550, 11.925),
        Point2(5.700, 11.925),
        Point2(5.700, 7.145),
        Point2(5.025, 7.145),
        Point2(5.025, 7.345),
        Point2(3.965, 7.345),
        Point2(3.965, 7.135),
        Point2(3.390, 7.135),
        Point2(3.390, 10.060),
        Point2(2.250, 10.060),
        Point2(2.250, 7.155),
        Point2(1.440, 7.155),
        Point2(1.440, 7.535),
        Point2(0.000, 7.535),
        Point2(0.000, 6.740),
        Point2(-0.195, 6.740),
        Point2(-0.195, 5.325),
        Point2(0.000, 5.325),
        Point2(0.000, 3.670),
        Point2(-0.195, 3.670),
        Point2(-0.195, 1.305),
        Point2(0.000, 1.305),
    ]


def lab_elevator_room_doorways() -> list[Doorway]:
    """
    :return: The doorway to the corridor in the right wall, and the doorway facing it
        in the recess of the left wall.
    """
    return [
        Doorway(start=Point2(9.5, 5.35), end=Point2(9.5, 6.63)),
        Doorway(start=Point2(-0.195, 5.35), end=Point2(-0.195, 6.63)),
    ]


def lab_elevator_room_staircase_edges() -> list[OpenEdge]:
    """
    :return: The outline edge between the left wall and the wall beside the elevator
        shaft, where a staircase leaves the room instead of a wall.
    """
    return [OpenEdge(start=Point2(1.440, 7.535), end=Point2(0.000, 7.535))]


@dataclass
class LabElevatorRoom:
    """
    World-independent description of the leftmost room of the lab occupancy grid map
    (``lab_map``, 0.05 m per cell): its walls and floor as traced from the map, its two
    doorways, the open side to the staircase left of the elevator, and an elevator
    filling the shaft in the middle of the top wall.
    """

    room: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_elevator_room_outline(),
            doorways=lab_elevator_room_doorways(),
            open_edges=lab_elevator_room_staircase_edges(),
        )
    )
    """
    The walls, floor, doorways and staircase side of the room.
    """

    elevator: SlidingDoorElevator = field(
        default_factory=lambda: SlidingDoorElevator(scale=Scale(2.6, 1.14, 2.2))
    )
    """
    The elevator cabin, sized to fill the shaft.
    """

    elevator_center: Point2 = field(default_factory=lambda: Point2(2.82, 8.76))
    """
    The centre of the elevator's footprint in the room outline's frame.
    """

    map_T_room: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            -23.4266, 34.6279, 0, yaw=np.deg2rad(-5.7)
        )
    )
    """
    Placement of the room outline's frame in the map frame of the occupancy grid.
    """

    @property
    def room_T_elevator(self) -> HomogeneousTransformationMatrix:
        """
        :return: Placement of the elevator cabin's centre in the room outline's frame,
            standing on the floor with its doors facing into the room.
        """
        center = self.elevator_center.to_np()
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            center[0], center[1], self.elevator.scale.z / 2, yaw=np.pi / 2
        )

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
    ) -> Room:
        """
        Materialize this room and its elevator in ``world``.

        :param world: The world the room is added to.
        :param name: Prefix for the names of every entity this room creates.
        :param parent: The entity standing for the map frame. If None, ``world.root`` is
            used.
        :return: The room annotation.
        """
        room = self.room.spawn(world, name, parent, parent_T_self=self.map_T_room)
        self.elevator.spawn(
            world,
            f"{name}_elevator",
            parent=room.floor.root,
            parent_T_self=self.room_T_elevator,
        )
        return room
