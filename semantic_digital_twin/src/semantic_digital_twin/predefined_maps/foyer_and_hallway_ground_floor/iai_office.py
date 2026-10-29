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
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    PlanarBoundingBox,
    Scale,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Level,
        Room,
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
    reached from them, as the floor plans of the building show them.

    Doorways are left open and windows are not modelled.
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
        :return: The smallest rectangle around every room of ``floor_plan``, axis-aligned
            in the outline frame of the elevator room.
        """
        room_T_map = self.elevator_room.map_T_room.inverse().to_np()
        corners = np.array(
            [
                room_T_map
                @ self.map_T_outline_frame(office_room.outline_frame).to_np()
                @ np.array([*vertex.to_np(), 0.0, 1.0])
                for office_room in floor_plan.rooms
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

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
    ) -> list[Level]:
        """
        Materialize the elevator room, its elevator and hallway, and the rooms and
        hallways of every storey in ``world``, with a level annotation for each storey.

        :param world: The world the office is added to.
        :param name: Prefix for the names of every entity the office creates.
        :param parent: The entity standing for the map frame. If None, ``world.root`` is
            used.
        :return: The level annotation of each storey, from the ground floor up.
        """
        levels = []
        with world.modify_world():
            elevator_room = self.elevator_room.spawn(world, name, parent)
            for floor_plan in sorted(self.floor_plans, key=lambda plan: plan.storey):
                self._spawn_rooms(world, name, parent, floor_plan)
                levels.append(self._spawn_level(world, name, elevator_room, floor_plan))
        return levels

    def _spawn_rooms(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None,
        floor_plan: FloorPlan,
    ) -> list[Room]:
        """
        Spawn every room of ``floor_plan`` at the elevation of its storey.

        :return: The room annotations.
        """
        ground_floor_T_storey = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=floor_plan.storey.elevation
        )
        return [
            office_room.layout.spawn(
                world,
                f"{name}_room_{office_room.number}",
                parent,
                parent_T_self=self.map_T_outline_frame(office_room.outline_frame)
                @ ground_floor_T_storey,
            )
            for office_room in floor_plan.rooms
        ]

    def _spawn_level(
        self, world: World, name: str, elevator_room: Room, floor_plan: FloorPlan
    ) -> Level:
        """
        Spawn the level of ``floor_plan``'s storey, spanning its footprint from the
        storey's floor up to the next storey's.

        :param elevator_room: The room annotation of the elevator room, whose floor the
            level attaches to.
        :return: The level annotation.
        """
        storey = floor_plan.storey
        footprint = self._footprint(floor_plan)
        return storey.level_type.get_annotation_specification(
            f"{name}_{storey.name.lower()}",
            storey.level_type.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(footprint.depth, footprint.width, Storey.height)
            ),
        ).spawn(
            world,
            parent=elevator_room.floor.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                (footprint.min_x + footprint.max_x) / 2,
                (footprint.min_y + footprint.max_y) / 2,
                storey.elevation + Storey.height / 2,
            ),
        )
