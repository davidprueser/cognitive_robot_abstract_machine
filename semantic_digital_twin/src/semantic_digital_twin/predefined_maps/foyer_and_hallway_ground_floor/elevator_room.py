from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    Doorway,
    OpenEdge,
    RectilinearRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.sliding_door_elevator import (
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

# %% ground floor


def lab_foyer_outline() -> list[Point2]:
    """
    :return: The inner wall faces of foyer 0.61 on the ground floor plan,
        counter-clockwise from its south-east corner, with x pointing north and y
        pointing west. The staircase and the elevator shaft take the foyer's south-west
        corner; the pocket between y 7.27 and 10.03 is the shaft around the cabin.
    """
    return [
        Point2(0.00, 0.00),
        Point2(9.62, 0.00),
        Point2(9.62, 12.05),
        Point2(5.69, 12.05),
        Point2(5.69, 7.27),
        Point2(3.47, 7.27),
        Point2(3.47, 10.03),
        Point2(2.27, 10.03),
        Point2(2.27, 7.27),
        Point2(1.52, 7.27),
        Point2(0.00, 7.27),
    ]


def lab_staircase_edge() -> OpenEdge:
    """
    :return: The outline edge between the elevator shaft and the south wall, where the
        staircase beside the shaft opens onto the foyer and onto the landing above it.
    """
    return OpenEdge(start=Point2(1.52, 7.27), end=Point2(0.00, 7.27))


# %% first floor


def lab_landing_outline() -> list[Point2]:
    """
    :return: The inner wall faces of landing Treppe 4 on the first floor plan, in the
        foyer outline's frame, counter-clockwise from its south-east corner. Between y
        7.27 and 7.71 it reaches into the elevator shaft up to the cabin's front.
    """
    return [
        Point2(0.00, 4.12),
        Point2(5.49, 4.12),
        Point2(5.49, 7.27),
        Point2(3.97, 7.27),
        Point2(3.47, 7.27),
        Point2(3.47, 7.71),
        Point2(2.27, 7.71),
        Point2(2.27, 7.27),
        Point2(1.52, 7.27),
        Point2(0.00, 7.27),
    ]


def lab_landing_open_edges() -> list[OpenEdge]:
    """
    :return: The landing's edges without a wall: the railing above the foyer, where the
        railing is left out, the staircase north and south of the elevator shaft, and
        the elevator entrance.
    """
    return [
        OpenEdge(start=Point2(0.00, 4.12), end=Point2(5.49, 4.12)),
        OpenEdge(start=Point2(5.49, 7.27), end=Point2(3.97, 7.27)),
        OpenEdge(start=Point2(3.47, 7.71), end=Point2(2.27, 7.71)),
        lab_staircase_edge(),
    ]


# %% hallway


def lab_hallway_outline() -> list[Point2]:
    """
    :return: The inner wall faces of the hallway on the ground floor plan, in the foyer
        outline's frame, counter-clockwise from the outer face of the foyer's north
        wall: corridor 0.63 running north from the foyer, and corridors 0.64 and 0.65
        running east from its end up to the stairwell TH4.
    """
    return [
        Point2(9.82, 5.13),
        Point2(24.60, 5.13),
        Point2(24.60, -37.13),
        Point2(26.28, -37.13),
        Point2(26.28, 6.82),
        Point2(9.82, 6.82),
    ]


# %% building


@dataclass
class LabElevatorRoom:
    """
    World-independent description of foyer 0.61 of the ECOTEC V building, as drawn in
    its floor plans: its walls and floor, the doorway to the hallway, the staircase
    opening onto it, an elevator filling the shaft in its west wall, landing Treppe 4 in
    front of the elevator on the first floor, looking down into the foyer, and the
    hallway beyond the doorway.

    The side rooms are not modelled; walls stand where their doors are.
    """

    @classproperty
    def first_floor_height(cls) -> float:
        """
        Height of the first floor's surface above the ground floor's, in metres.
        """
        return 3.45

    room: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_foyer_outline(),
            doorways=[Doorway(start=Point2(9.62, 5.39), end=Point2(9.62, 6.72))],
            open_edges=[lab_staircase_edge()],
            wall_height=LabElevatorRoom.first_floor_height,
            wall_thickness=0.2,
        )
    )
    """
    The walls, floor, hallway doorway and staircase side of the foyer on the ground
    floor.

    The walls reach up to the first floor.
    """

    first_floor: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_landing_outline(),
            open_edges=lab_landing_open_edges(),
            wall_thickness=0.2,
        )
    )
    """
    The landing in front of the elevator on the first floor, in the ground floor
    outline's frame.
    """

    hallway: RectilinearRoom = field(
        default_factory=lambda: RectilinearRoom(
            outline=lab_hallway_outline(),
            open_edges=[OpenEdge(start=Point2(9.82, 5.13), end=Point2(9.82, 6.82))],
            wall_height=LabElevatorRoom.first_floor_height,
        )
    )
    """
    The hallway beyond the foyer's doorway, in the ground floor outline's frame, open
    towards the foyer.
    """

    elevator: SlidingDoorElevator = field(
        default_factory=lambda: SlidingDoorElevator(scale=Scale(2.31, 1.2, 2.2))
    )
    """
    The elevator cabin.
    """

    elevator_center: Point2 = field(default_factory=lambda: Point2(2.87, 8.87))
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
        Materialize the foyer, its landing on the first floor, its elevator and the
        hallway in ``world``, with a level annotation for each floor the elevator
        serves.

        :param world: The world the foyer is added to.
        :param name: Prefix for the names of every entity this creates.
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
        self.hallway.spawn(world, f"{name}_hallway", parent=room.floor.root)
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
