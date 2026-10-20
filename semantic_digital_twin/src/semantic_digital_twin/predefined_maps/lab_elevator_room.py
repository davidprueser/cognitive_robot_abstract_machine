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
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    FirstFloor,
    GroundFloor,
    Level,
    Room,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

FIRST_FLOOR_HEIGHT = 4.5
"""
Height of the first floor's surface above the ground floor's, in metres.
"""

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
    :return: The doorway to the corridor in the right wall. The first floor has a door
        right above it.
    """
    return Doorway(start=Point2(9.5, 5.35), end=Point2(9.5, 6.63))


def lab_left_wall_doorway() -> Doorway:
    """
    :return: The doorway in the recess of the left wall, facing the corridor doorway.
        The first floor has a door right above it.
    """
    return Doorway(start=Point2(-0.195, 5.35), end=Point2(-0.195, 6.63))


def lab_elevator_room_doorways() -> list[Doorway]:
    """
    :return: The doorway to the corridor in the right wall, and the doorway facing it
        in the recess of the left wall. The first floor has a door right above each.
    """
    return [lab_corridor_doorway(), lab_left_wall_doorway()]


def lab_elevator_room_staircase_edges() -> list[OpenEdge]:
    """
    :return: The outline edge between the left wall and the wall beside the elevator
        shaft, where a staircase leaves the room instead of a wall.
    """
    return [OpenEdge(start=Point2(1.440, 7.535), end=Point2(0.000, 7.535))]


# %% first floor


def lab_first_floor_outline() -> list[Point2]:
    """
    :return: The inner wall faces of the first-floor walkway, in the ground floor
        outline's frame, counter-clockwise from its front left corner. The walkway runs
        along the ground floor's top wall from the left wall to the right wall, and
        reaches 0.3 m further into the room than the doors above the ground floor's
        doorways need.
        Between x 2.25 and 3.39 it reaches into the elevator shaft up to the cabin's
        front.
    """
    return [
        Point2(0.000, 5.025),
        Point2(9.500, 5.025),
        Point2(9.500, 7.145),
        Point2(5.700, 7.145),
        Point2(5.025, 7.145),
        Point2(5.025, 7.345),
        Point2(3.965, 7.345),
        Point2(3.965, 7.135),
        Point2(3.390, 7.135),
        Point2(3.390, 7.460),
        Point2(2.250, 7.460),
        Point2(2.250, 7.155),
        Point2(1.440, 7.155),
        Point2(1.440, 7.535),
        Point2(0.000, 7.535),
        Point2(0.000, 6.740),
        Point2(-0.195, 6.740),
        Point2(-0.195, 5.325),
        Point2(0.000, 5.325),
    ]


def lab_first_floor_open_edges() -> list[OpenEdge]:
    """
    :return: The walkway's edges without a wall: the side facing the ground floor and
        the side facing the room's top right part, where the railing is left out, the
        elevator entrance, and the staircase.
    """
    return [
        OpenEdge(start=Point2(0.000, 5.025), end=Point2(9.500, 5.025)),
        OpenEdge(start=Point2(9.500, 7.145), end=Point2(5.700, 7.145)),
        OpenEdge(start=Point2(3.390, 7.460), end=Point2(2.250, 7.460)),
        *lab_elevator_room_staircase_edges(),
    ]


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
        the room and towards the south stretch, with the doors of two side rooms in its
        south wall.
    """
    return RectilinearRoom(
        outline=lab_hallway_east_stretch_outline(),
        doorways=[
            Doorway(start=Point2(-13.35, 37.2), end=Point2(-12.55, 37.2)),
            Doorway(start=Point2(-7.55, 37.2), end=Point2(-6.85, 37.2)),
        ],
        open_edges=[
            OpenEdge(start=Point2(-17.48, 38.8), end=Point2(-17.48, 37.2)),
            OpenEdge(start=Point2(-2.48, 37.2), end=Point2(-0.93, 37.2)),
        ],
        wall_height=FIRST_FLOOR_HEIGHT,
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
        with the doors of three side rooms and, across its whole south end, the doorway
        into the room there.
    """
    return RectilinearRoom(
        outline=lab_hallway_south_stretch_outline(),
        doorways=[
            Doorway(start=Point2(-3.0, -5.1), end=Point2(-1.45, -5.1)),
            Doorway(start=Point2(-1.45, 3.1), end=Point2(-1.45, 4.1)),
            Doorway(start=Point2(-0.62, 18.25), end=Point2(-0.62, 18.95)),
            Doorway(start=Point2(-3.65, 19.9), end=Point2(-3.65, 20.6)),
        ],
        open_edges=[OpenEdge(start=Point2(-1.45, 37.17), end=Point2(-3.0, 37.17))],
        wall_height=FIRST_FLOOR_HEIGHT,
    )


@dataclass
class LabHallway:
    """
    World-independent description of the hallway leading away from the corridor doorway
    of the lab elevator room, as traced from the lab occupancy grid map.

    It runs east and then a long way south, and is split into one straight stretch per
    direction because the map bends slightly between them. The side rooms are not
    modelled, only their doors.
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

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
    ) -> list[Room]:
        """
        Materialize both stretches of the hallway in ``world``.

        :param world: The world the hallway is added to.
        :param name: Prefix for the names of every entity the hallway creates.
        :param parent: The entity standing for the map frame. If None, ``world.root`` is
            used.
        :return: The room annotations of the east and the south stretch.
        """
        return [
            self.east_stretch.spawn(
                world, f"{name}_east", parent, parent_T_self=self.map_T_east_stretch
            ),
            self.south_stretch.spawn(
                world, f"{name}_south", parent, parent_T_self=self.map_T_south_stretch
            ),
        ]


# %% room


@dataclass
class LabElevatorRoom:
    """
    World-independent description of the leftmost room of the lab occupancy grid map
    (``lab_map``, 0.05 m per cell): its walls and floor as traced from the map, its two
    doorways, the open side to the staircase left of the elevator, an elevator filling
    the shaft in the middle of the top wall, the first-floor walkway in front of the
    elevator, looking down into the room, and the hallway beyond the corridor doorway.
    """

    room: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_elevator_room_outline(),
            doorways=lab_elevator_room_doorways(),
            open_edges=lab_elevator_room_staircase_edges(),
            wall_height=FIRST_FLOOR_HEIGHT,
        )
    )
    """
    The walls, floor, doorways and staircase side of the room on the ground floor.

    The walls reach up to the first floor.
    """

    first_floor: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_first_floor_outline(),
            doorways=lab_elevator_room_doorways(),
            open_edges=lab_first_floor_open_edges(),
        )
    )
    """
    The walkway on the first floor, in the ground floor outline's frame.
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
    def room_T_first_floor(self) -> HomogeneousTransformationMatrix:
        """
        :return: Placement of the first floor's outline frame in the ground floor's,
            resting on top of the ground floor's walls.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(z=self.room.wall_height)

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
        Materialize this room, its first-floor walkway, its elevator and the hallway in
        ``world``, with a level annotation for each floor the elevator serves.

        :param world: The world the room is added to.
        :param name: Prefix for the names of every entity this room creates.
        :param parent: The entity standing for the map frame. If None, ``world.root`` is
            used.
        :return: The ground floor room annotation.
        """
        room = self.room.spawn(world, name, parent, parent_T_self=self.map_T_room)
        first_floor = self.first_floor.spawn(
            world,
            f"{name}_first_storey",
            parent=room.floor.root,
            parent_T_self=self.room_T_first_floor,
        )
        self.hallway.spawn(world, f"{name}_hallway", parent)
        self.elevator.spawn(
            world,
            f"{name}_elevator",
            parent=room.floor.root,
            parent_T_self=self.room_T_elevator,
        )
        self._spawn_level(world, GroundFloor, f"{name}_ground_floor", self.room, room)
        self._spawn_level(
            world, FirstFloor, f"{name}_first_floor", self.first_floor, first_floor
        )
        return room

    @staticmethod
    def _spawn_level(
        world: World,
        level_type: type[Level],
        name: str,
        description: RectilinearRoom,
        room: Room,
    ) -> Level:
        """
        Spawn a level whose region spans ``room`` from its floor up to the top of its
        walls.

        :param level_type: The level annotation to spawn.
        :param description: The description ``room`` was spawned from.
        :return: The level annotation.
        """
        footprint = description.footprint
        return level_type.get_annotation_specification(
            name,
            level_type.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(footprint.depth, footprint.width, description.wall_height)
            ),
        ).spawn(
            world,
            parent=room.floor.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                (footprint.min_x + footprint.max_x) / 2,
                (footprint.min_y + footprint.max_y) / 2,
                description.wall_height / 2,
            ),
        )
