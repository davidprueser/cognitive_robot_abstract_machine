from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.elevator_room import (
    LabElevatorRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
    FloorPlan,
    FloorPlanFile,
    OutlineFrame,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey_layout import (
    PlacedRoom,
    StoreyLayout,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    PlanarBoundingBox,
    Scale,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Floor,
        Level,
    )
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

# %% office


@dataclass
class IAIOffice:
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
        default_factory=lambda: [FloorPlan.read(file) for file in FloorPlanFile]
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
                    f"room_{office_room.number}",
                    office_room.layout,
                    self.map_T_outline_frame(office_room.outline_frame),
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
    ) -> list[Level]:
        """
        Materialize every storey of the office in ``world``, with a level annotation
        each, and the elevator serving them.

        :param world: The world the office is added to.
        :param name: Prefix for the names of every entity the office creates.
        :param parent: The entity standing for the map frame. If None, ``world.root`` is
            used.
        :return: The level annotation of each storey, from the ground floor up.
        """
        levels = []
        with world.modify_world():
            for floor_plan in sorted(self.floor_plans, key=lambda plan: plan.storey):
                storey = floor_plan.storey
                storey_name = f"{name}_{storey.name.lower()}"
                hallway = self.storey_layout(floor_plan).spawn(
                    world,
                    storey_name,
                    parent,
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=storey.elevation
                    ),
                )
                levels.append(
                    self._spawn_level(world, storey_name, hallway.floor, floor_plan)
                )
            self.elevator_room.elevator.spawn(
                world,
                f"{name}_elevator",
                parent=parent,
                parent_T_self=self.elevator_room.map_T_room
                @ self.elevator_room.room_T_elevator,
            )
        return levels

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
