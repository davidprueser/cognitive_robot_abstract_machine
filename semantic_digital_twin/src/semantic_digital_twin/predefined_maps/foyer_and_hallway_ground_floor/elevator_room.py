from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    Doorway,
    OpenEdge,
    RectilinearRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.sliding_door_elevator import (
    SlidingDoorElevator,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world_description.geometry import Scale

# %% ground floor


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


def lab_corridor_doorway() -> Doorway:
    """
    :return: The doorway to the corridor in the right wall.
    """
    return Doorway(start=Point2(9.5, 5.35), end=Point2(9.5, 6.63))


def lab_left_wall_doorway() -> Doorway:
    """
    :return: The doorway in the recess of the left wall, facing the corridor doorway.
    """
    return Doorway(start=Point2(-0.195, 5.35), end=Point2(-0.195, 6.63))


def lab_elevator_room_doorways() -> list[Doorway]:
    """
    :return: The doorway to the corridor in the right wall, and the doorway facing it
        in the recess of the left wall.
    """
    return [lab_corridor_doorway(), lab_left_wall_doorway()]


def lab_elevator_room_staircase_edges() -> list[OpenEdge]:
    """
    :return: The outline edge between the left wall and the wall beside the elevator
        shaft, where a staircase leaves the room instead of a wall.
    """
    return [OpenEdge(start=Point2(1.440, 7.535), end=Point2(0.000, 7.535))]


# %% hallway


def lab_hallway_east_stretch_outline() -> list[Point2]:
    """
    :return: The inner wall faces of the hallway's stretch running east from the
        corridor doorway, in the map frame turned by the stretch's yaw,
        counter-clockwise from the outer face of the room's right wall. The stretch
        holds a niche in its north wall and two pillars, and the south stretch leaves it
        between x -2.48 and -0.93.
    """
    return [
        Point2(-17.480, 37.200),
        Point2(-17.100, 37.200),
        Point2(-17.100, 37.000),
        Point2(-16.250, 37.000),
        Point2(-16.250, 37.200),
        Point2(-12.200, 37.200),
        Point2(-12.200, 37.420),
        Point2(-11.850, 37.420),
        Point2(-11.850, 37.200),
        Point2(-2.480, 37.200),
        Point2(-0.930, 37.200),
        Point2(-0.930, 37.370),
        Point2(-0.150, 37.370),
        Point2(-0.150, 38.800),
        Point2(-7.450, 38.800),
        Point2(-7.450, 39.500),
        Point2(-11.200, 39.500),
        Point2(-11.200, 38.800),
        Point2(-17.480, 38.800),
    ]


def lab_hallway_east_stretch() -> RectilinearRoom:
    """
    :return: The hallway's stretch running east from the corridor doorway, open towards
        the south stretch, with the doorways of the side rooms the
        floor plan of the ground floor shows along it.
    """
    return RectilinearRoom(
        outline=lab_hallway_east_stretch_outline(),
        doorways=[
            Doorway(start=Point2(-14.717, 37.2), end=Point2(-13.716, 37.2)),
            Doorway(start=Point2(-13.32, 37.2), end=Point2(-12.431, 37.2)),
            Doorway(start=Point2(-10.918, 37.2), end=Point2(-10.032, 37.2)),
            Doorway(start=Point2(-8.887, 37.2), end=Point2(-8.006, 37.2)),
            Doorway(start=Point2(-7.55, 37.2), end=Point2(-6.664, 37.2)),
            Doorway(start=Point2(-0.15, 37.485), end=Point2(-0.15, 38.746)),
            Doorway(start=Point2(-14.692, 38.8), end=Point2(-13.572, 38.8)),
            Doorway(start=Point2(-10.442, 39.5), end=Point2(-9.447, 39.5)),
            Doorway(start=Point2(-9.059, 39.5), end=Point2(-8.061, 39.5)),
        ],
        open_edges=[OpenEdge(start=Point2(-2.48, 37.2), end=Point2(-0.93, 37.2))],
        wall_height=Storey.height,
    )


def lab_hallway_south_stretch_outline() -> list[Point2]:
    """
    :return: The inner wall faces of the hallway's long stretch running south from the
        east stretch, in the map frame turned by the stretch's yaw, counter-clockwise
        from the west end of the doorway into the room at its south end. The stretch
        widens into a lobby between y 17.7 and 21.1 and holds a niche on either side.
    """
    return [
        Point2(-3.00, -5.10),
        Point2(-1.45, -5.10),
        Point2(-1.45, 17.70),
        Point2(-0.62, 17.70),
        Point2(-0.62, 21.10),
        Point2(-1.45, 21.10),
        Point2(-1.45, 27.80),
        Point2(-0.65, 27.80),
        Point2(-0.65, 29.40),
        Point2(-1.45, 29.40),
        Point2(-1.45, 37.17),
        Point2(-3.00, 37.17),
        Point2(-3.00, 21.10),
        Point2(-3.65, 21.10),
        Point2(-3.65, 17.70),
        Point2(-3.00, 17.70),
        Point2(-3.00, 4.60),
        Point2(-3.72, 4.60),
        Point2(-3.72, 1.20),
        Point2(-3.00, 1.20),
    ]


def lab_hallway_south_stretch() -> RectilinearRoom:
    """
    :return: The hallway's long stretch running south, open towards the east stretch,
        with, across its whole south end, the doorway into the staircase there, and the
        doorways of the side rooms the floor plan of the ground floor shows along it.
    """
    return RectilinearRoom(
        outline=lab_hallway_south_stretch_outline(),
        doorways=[
            Doorway(start=Point2(-3.0, -5.1), end=Point2(-1.45, -5.1)),
            Doorway(start=Point2(-3.72, 1.776), end=Point2(-3.72, 2.783)),
            Doorway(start=Point2(-3.72, 3.153), end=Point2(-3.72, 4.58)),
            Doorway(start=Point2(-3.65, 18.273), end=Point2(-3.65, 19.278)),
            Doorway(start=Point2(-3.65, 19.65), end=Point2(-3.65, 21.08)),
            Doorway(start=Point2(-3.0, 25.046), end=Point2(-3.0, 25.945)),
            Doorway(start=Point2(-3.0, 26.299), end=Point2(-3.0, 27.194)),
            Doorway(start=Point2(-3.0, 28.707), end=Point2(-3.0, 29.601)),
            Doorway(start=Point2(-3.0, 30.823), end=Point2(-3.0, 31.714)),
            Doorway(start=Point2(-1.45, 1.269), end=Point2(-1.45, 2.783)),
            Doorway(start=Point2(-1.45, 3.138), end=Point2(-1.45, 4.651)),
            Doorway(start=Point2(-1.45, 33.587), end=Point2(-1.45, 34.474)),
            Doorway(start=Point2(-0.65, 27.89), end=Point2(-0.65, 28.897)),
            Doorway(start=Point2(-0.62, 17.767), end=Point2(-0.62, 19.278)),
            Doorway(start=Point2(-0.62, 19.65), end=Point2(-0.62, 21.08)),
        ],
        open_edges=[OpenEdge(start=Point2(-1.45, 37.17), end=Point2(-3.0, 37.17))],
        wall_height=Storey.height,
    )


@dataclass
class LabHallway:
    """
    World-independent description of the hallway leading away from the corridor doorway
    of the lab elevator room, as traced from the lab occupancy grid map.

    It runs east and then a long way south, and is split into one straight stretch per
    direction because the map bends slightly between them.
    """

    east_stretch: RectilinearRoom = field(default_factory=lab_hallway_east_stretch)
    """
    The stretch running east from the corridor doorway.
    """

    south_stretch: RectilinearRoom = field(default_factory=lab_hallway_south_stretch)
    """
    The long stretch running south.
    """

    map_T_east_stretch: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            yaw=np.deg2rad(-6.2)
        )
    )
    """
    Placement of the east stretch's outline frame in the map frame.
    """

    map_T_south_stretch: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            yaw=np.deg2rad(-7.0)
        )
    )
    """
    Placement of the south stretch's outline frame in the map frame.
    """


# %% room


@dataclass
class LabElevatorRoom:
    """
    World-independent description of the leftmost room of the lab occupancy grid map
    (``lab_map``, 0.05 m per cell): its walls and floor as traced from the map, its two
    doorways, the open side to the staircase left of the elevator, an elevator filling
    the shaft in the middle of the top wall, and the hallway beyond the corridor
    doorway.
    """

    room: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_elevator_room_outline(),
            doorways=lab_elevator_room_doorways(),
            open_edges=lab_elevator_room_staircase_edges(),
            wall_height=Storey.height,
        )
    )
    """
    The walls, floor, doorways and staircase side of the room on the ground floor.

    The walls reach up to the first floor.
    """

    hallway: LabHallway = field(default_factory=LabHallway)
    """
    The hallway beyond the corridor doorway.
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
