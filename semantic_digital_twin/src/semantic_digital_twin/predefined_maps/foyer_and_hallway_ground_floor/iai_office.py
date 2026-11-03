from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.elevator_room import (
    LabElevatorRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
    FloorPlan,
    OutlineFrame,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plans.first_floor import (
    FirstFloorPlan,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plans.ground_floor import (
    GroundFloorPlan,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plans.second_floor import (
    SecondFloorPlan,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey_layout import (
    PlacedRoom,
    StoreyLayout,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    ElevatorLobby,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    PlanarBoundingBox,
    Scale,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Elevator,
        Floor,
        Level,
        Room,
    )
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

# %% office


@dataclass
class IAIOffice(ElevatorBuilding):
    """
    World-independent description of the IAI's office: the lab elevator room with its
    hallway, and the rooms and hallways on the ground, first and second floor that are
    reached from them, as the floor plans of the building show them. Each storey stands
    on one floor.

    Doorways are left open, without doors, and windows are not modelled.
    """

    elevator_room: LabElevatorRoom = field(default_factory=LabElevatorRoom)
    """
    The lab elevator room, its elevator and the hallway on the ground floor.
    """

    floor_plans: list[FloorPlan] = field(
        default_factory=lambda: [
            GroundFloorPlan(),
            FirstFloorPlan(),
            SecondFloorPlan(),
        ]
    )
    """
    The rooms and hallways of each storey around the elevator room and its hallway.
    """

    def map_T_outline_frame(
        self, outline_frame: OutlineFrame
    ) -> HomogeneousTransformationMatrix:
        """
        :param outline_frame: The frame an office room's outline is given in.
        :return: Placement of that frame in the map frame, at ground floor level.
        """
        match outline_frame:
            case OutlineFrame.ELEVATOR_ROOM:
                return self.elevator_room.map_T_room
            case OutlineFrame.HALLWAY_EAST_STRETCH:
                return self.elevator_room.hallway.map_T_east_stretch
            case OutlineFrame.HALLWAY_SOUTH_STRETCH:
                return self.elevator_room.hallway.map_T_south_stretch

    def _footprint(self, floor_plan: FloorPlan) -> PlanarBoundingBox:
        """
        :param floor_plan: One of :attr:`floor_plans`.
        :return: The smallest rectangle around the rooms and the hallway of
            ``floor_plan``, axis-aligned
            in the outline frame of the elevator room.
        """
        room_T_map = self.elevator_room.map_T_room.inverse().to_np()
        corners = np.array(
            [
                room_T_map
                @ self.map_T_outline_frame(office_room.outline_frame).to_np()
                @ np.array([*vertex.to_np(), 0.0, 1.0])
                for office_room in floor_plan.rooms + floor_plan.hallway
                for vertex in office_room.layout.outline
            ]
        )
        minimum_x, minimum_y = corners[:, :2].min(axis=0)
        maximum_x, maximum_y = corners[:, :2].max(axis=0)
        return PlanarBoundingBox(
            float(minimum_x),
            float(minimum_y),
            float(maximum_x),
            float(maximum_y),
            HomogeneousTransformationMatrix(),
        )

    def storey_layout(self, floor_plan: FloorPlan) -> StoreyLayout:
        """
        :param floor_plan: One of :attr:`floor_plans`.
        :return: The rooms and the hallway of ``floor_plan``'s storey placed in the map
            frame, together with the elevator room and its hallway on the ground floor.
        """
        rooms, hallway = (
            [
                PlacedRoom(
                    self.room_name(office_room.number),
                    office_room.layout,
                    self.map_T_outline_frame(office_room.outline_frame),
                    room_type=office_room.room_type,
                    cabin_names=[
                        self.room_name(cabin_number)
                        for cabin_number in office_room.cabin_numbers
                    ],
                )
                for office_room in office_rooms
            ]
            for office_rooms in (floor_plan.rooms, floor_plan.hallway)
        )
        if floor_plan.storey is Storey.GROUND_FLOOR:
            lab_hallway = self.elevator_room.hallway
            rooms.append(
                PlacedRoom(
                    "elevator_room",
                    self.elevator_room.room,
                    self.elevator_room.map_T_room,
                    room_type=ElevatorLobby,
                )
            )
            hallway += [
                PlacedRoom(
                    "hallway_east",
                    lab_hallway.east_stretch,
                    lab_hallway.map_T_east_stretch,
                ),
                PlacedRoom(
                    "hallway_south",
                    lab_hallway.south_stretch,
                    lab_hallway.map_T_south_stretch,
                ),
            ]
        return StoreyLayout(rooms, hallway)

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
    ) -> None:
        """
        Every storey of the office, with a level annotation each, and the elevator
        serving them.

        :param parent: The entity standing for the map frame. If None, ``world.root`` is
            used.
        """
        with world.modify_world():
            for floor_plan in sorted(self.floor_plans, key=lambda plan: plan.storey):
                storey_name = self.storey_name(name, floor_plan.storey)
                hallway = self.storey_layout(floor_plan).spawn(
                    world,
                    storey_name,
                    parent,
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=floor_plan.storey.elevation
                    ),
                )
                self._spawn_level(world, storey_name, hallway.floor, floor_plan)
            self.elevator_room.elevator.spawn(
                world,
                f"{name}_elevator",
                parent=parent,
                parent_T_self=self.elevator_room.map_T_room
                @ self.elevator_room.room_T_elevator,
            )

    def navigation_targets(self, world: World, name: str) -> list[Room]:
        """
        The rooms of every floor plan that are marked as
        :attr:`~semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan.OfficeRoom.navigation_target`.
        """
        return [
            self.get_room(world, name, floor_plan.storey, office_room.number)
            for floor_plan in self.floor_plans
            for office_room in floor_plan.rooms
            if office_room.navigation_target
        ]

    def simulated_drive_joint_name(self, elevator: Elevator) -> str:
        """
        The name of ``elevator``'s drive connection in the world, which the simulator's
        world is exported with.
        """
        return elevator.movable_joint.name.name

    def get_room(self, world: World, name: str, storey: Storey, number: str) -> Room:
        """
        :param world: The world the office was spawned into.
        :param name: The prefix the office was spawned with.
        :param storey: The storey the room lies on.
        :param number: The room's number in its floor plan.
        :return: The room's annotation.
        """
        return world.get_semantic_annotation_by_name(
            f"{self.storey_name(name, storey)}_{self.room_name(number)}"
        )

    @staticmethod
    def room_name(number: str) -> str:
        """
        :param number: A room's number in its floor plan.
        :return: The name the room is placed under in its storey.
        """
        return f"room_{number}"

    @staticmethod
    def storey_name(name: str, storey: Storey) -> str:
        """
        :param name: The prefix the office is spawned with.
        :param storey: One of the office's storeys.
        :return: The prefix of the names of every entity ``storey`` is spawned with.
        """
        return f"{name}_{storey.name.lower()}"

    def _spawn_level(
        self, world: World, name: str, floor: Floor, floor_plan: FloorPlan
    ) -> Level:
        """
        Spawn the level of ``floor_plan``'s storey, spanning its footprint from the
        storey's floor up to the next storey's.

        :param name: Name of the level.
        :param floor: The hallway floor of the storey, whose frame is the map frame at
            the storey's elevation.
        :return: The level annotation.
        """
        storey = floor_plan.storey
        footprint = self._footprint(floor_plan)
        return storey.level_type.get_annotation_specification(
            name,
            storey.level_type.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(footprint.depth, footprint.width, Storey.height)
            ),
        ).spawn(
            world,
            parent=floor.root,
            parent_T_self=self.elevator_room.map_T_room
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                (footprint.min_x + footprint.max_x) / 2,
                (footprint.min_y + footprint.max_y) / 2,
                Storey.height / 2,
            ),
        )
