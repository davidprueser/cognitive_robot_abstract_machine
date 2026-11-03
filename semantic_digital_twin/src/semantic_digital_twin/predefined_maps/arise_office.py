from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Level,
    Room,
    Wall,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    PlanarBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Region

# %% the office of the Gazebo simulator


class WallSide(StrEnum):
    """
    The edge of a storey an outer wall stands on, by compass direction with north
    towards +y.
    """

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class AriseOfficeRoom(StrEnum):
    """
    A room of a storey of the office, named for where it lies relative to the elevator
    shaft.
    """

    WEST_OFFICE = "west_office"
    """
    The room west of the shaft, spanning the storey's whole depth.
    """

    LOBBY = "lobby"
    """
    The room on the ground floor between the offices, holding the elevator cabin.
    """

    LANDING = "landing"
    """
    The room on the first floor between the offices, in front of the shaft.
    """

    EAST_OFFICE = "east_office"
    """
    The room east of the shaft, spanning the storey's whole depth.
    """


@dataclass
class AriseOfficeBuilding(ElevatorBuilding):
    """
    World-independent description of the two-storey office of the TIAGo ARISE Gazebo
    simulator (``office_area.sdf`` and ``elevator.urdf`` of the
    ``tiago_arise_navigation`` package).

    The world root is the Gazebo world frame. The elevator's doors face -y.

    .. note:: The Gazebo elevator joint is 0 when the cabin serves the ground floor, so
        it equals the semantic drive position minus :attr:`ground_floor_height`.
    """

    storey_scale: Scale = field(default_factory=lambda: Scale(16.0, 8.0, 4.0))
    """
    Extents of each storey around the world origin and the height of its walls.
    """

    ground_floor_height: float = 0.1
    """
    Height of the top of the ground floor slab.
    """

    first_floor_height: float = 4.15
    """
    Height of the top of the first floor slab.
    """

    wall_thickness: float = 0.1
    """
    Thickness of the outer walls.
    """

    cabin_centre: Point3 = field(default_factory=lambda: Point3(0.0, 2.925, 0.0))
    """
    Centre of the elevator cabin's footprint in the world frame; its height is ignored.
    """

    cabin_scale: Scale = field(default_factory=lambda: Scale(1.6, 1.6, 2.0))
    """
    Depth from the doors to the back wall, width and height of the elevator cabin.
    """

    cabin_wall_thickness: float = 0.05
    """
    Thickness of the cabin's walls.
    """

    cabin_door_thickness: float = 0.04
    """
    Thickness of each of the two sliding cabin doors.
    """

    cabin_maximum_speed: float = 0.5
    """
    Maximum speed of the elevator drive, as limited in ``elevator.urdf``.
    """

    shaft_opening_width: float = 1.85
    """
    Width of the gap the first floor slab leaves for the shaft.
    """

    shaft_front: float = 2.125
    """
    Y at which the first floor slab in front of the shaft ends.
    """

    @classproperty
    def middle_rooms(cls) -> dict[type[Level], AriseOfficeRoom]:
        """
        The room between the two offices of each storey.
        """
        return {GroundFloor: AriseOfficeRoom.LOBBY, FirstFloor: AriseOfficeRoom.LANDING}

    def spawn(self, world: World, name: str = "arise_office") -> None:
        """
        Materialize the office in ``world``: the rooms and outer walls of both storeys,
        a level annotation per storey, and the elevator serving the ground floor.

        Each storey is tiled by an office on either side of the shaft and a room between
        them; on the first floor that room ends at the shaft.
        """
        self._spawn_storey(
            world,
            name,
            GroundFloor,
            self.ground_floor_height,
            self.storey_scale.y / 2,
        )
        self._spawn_storey(
            world,
            name,
            FirstFloor,
            self.first_floor_height,
            self.shaft_front,
        )
        self._spawn_elevator(world, f"{name}_elevator")

    def navigation_targets(
        self, world: World, name: str = "arise_office"
    ) -> list[Room]:
        """
        Every room of both storeys.
        """
        return [
            self.get_room(world, level_type, room, name)
            for level_type, middle_room in self.middle_rooms.items()
            for room in (
                AriseOfficeRoom.WEST_OFFICE,
                middle_room,
                AriseOfficeRoom.EAST_OFFICE,
            )
        ]

    def simulated_drive_joint_name(self, elevator: Elevator) -> str:
        """
        The name ``elevator.urdf`` gives the drive joint.
        """
        return "elevator_vertical_joint"

    def get_room(
        self,
        world: World,
        level_type: type[Level],
        room: AriseOfficeRoom,
        name: str = "arise_office",
    ) -> Room:
        """
        :param world: The world the office was spawned into.
        :param level_type: The storey the room lies on.
        :param room: Which room of the storey it is.
        :param name: The prefix the office was spawned with.
        :return: The room annotation, whose floor's frame lies on its top.
        """
        return world.get_semantic_annotation_by_name(
            self.room_name(name, level_type, room)
        )

    @staticmethod
    def room_name(name: str, level_type: type[Level], room: AriseOfficeRoom) -> str:
        """
        :param name: The prefix the office is spawned with.
        :param level_type: The storey the room lies on.
        :param room: Which room of the storey it is.
        :return: The name of the room annotation.
        """
        return f"{name}_{level_type.__name__}_{room}"

    # %% storeys

    def _spawn_storey(
        self,
        world: World,
        name: str,
        level_type: type[Level],
        height: float,
        middle_room_north_edge: float,
    ) -> None:
        """
        The rooms, outer walls and level of the storey whose floor top is at ``height``.

        :param name: The prefix the office is spawned with.
        :param middle_room_north_edge: Y at which the room between the offices ends.
        """
        half_x, half_y = self.storey_scale.x / 2, self.storey_scale.y / 2
        half_shaft = self.shaft_opening_width / 2
        origin = HomogeneousTransformationMatrix()
        for room, footprint in (
            (
                AriseOfficeRoom.WEST_OFFICE,
                PlanarBoundingBox(-half_x, -half_y, -half_shaft, half_y, origin),
            ),
            (
                self.middle_rooms[level_type],
                PlanarBoundingBox(
                    -half_shaft, -half_y, half_shaft, middle_room_north_edge, origin
                ),
            ),
            (
                AriseOfficeRoom.EAST_OFFICE,
                PlanarBoundingBox(half_shaft, -half_y, half_x, half_y, origin),
            ),
        ):
            self._spawn_room(
                world, self.room_name(name, level_type, room), footprint, height
            )
        storey_name = f"{name}_{level_type.__name__}"
        self._spawn_outer_walls(world, storey_name, height)
        self._spawn_level(world, f"{storey_name}_level", level_type, height)

    @staticmethod
    def _spawn_room(
        world: World, name: str, footprint: PlanarBoundingBox, height: float
    ) -> Room:
        """
        A room standing on a flat floor of its own, whose top is at ``height``.

        :param footprint: The room's extent in the world frame.
        """
        floor = Floor.get_annotation_specification(
            f"{name}_floor",
            Floor.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(footprint.depth, footprint.width, 0)
            ),
        ).spawn(
            world,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                footprint.x_interval.center(), footprint.y_interval.center(), height
            ),
        )
        room = Room(name=PrefixedName(name), floor=floor)
        with world.modify_world():
            world.add_semantic_annotation(room)
        return room

    def _spawn_outer_walls(self, world: World, name: str, height: float) -> None:
        """
        A wall along each edge of the storey whose floor top is at ``height``.
        """
        half_x, half_y = self.storey_scale.x / 2, self.storey_scale.y / 2
        for side, world_T_wall, length in (
            (
                WallSide.NORTH,
                HomogeneousTransformationMatrix.from_xyz_rpy(0, half_y, yaw=np.pi / 2),
                self.storey_scale.x,
            ),
            (
                WallSide.SOUTH,
                HomogeneousTransformationMatrix.from_xyz_rpy(0, -half_y, yaw=np.pi / 2),
                self.storey_scale.x,
            ),
            (
                WallSide.EAST,
                HomogeneousTransformationMatrix.from_xyz_rpy(half_x, 0),
                self.storey_scale.y,
            ),
            (
                WallSide.WEST,
                HomogeneousTransformationMatrix.from_xyz_rpy(-half_x, 0),
                self.storey_scale.y,
            ),
        ):
            Wall.get_annotation_specification(
                f"{name}_wall_{side}",
                Wall.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(self.wall_thickness, length, self.storey_scale.z)
                ),
            ).spawn(
                world,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=height + self.storey_scale.z / 2
                )
                @ world_T_wall,
            )

    def _spawn_level(
        self, world: World, name: str, level_type: type[Level], height: float
    ) -> None:
        """
        A region spanning one storey from the top of its floor up, annotated as a level.
        """
        with world.modify_world():
            region = Region(
                name=PrefixedName(name),
                area=ShapeCollection(shapes=[Box(scale=self.storey_scale)]),
            )
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=region,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=height + self.storey_scale.z / 2
                    ),
                )
            )
            world.add_semantic_annotation(level_type(root=region))

    # %% elevator

    def _spawn_elevator(self, world: World, name: str) -> None:
        """
        The cabin on a vertical drive with two sliding doors in its opening, built like
        the elevator of the multi-storey test building.

        The cabin's x-axis points from its doors into the cabin.
        """
        elevator_world = World.create_with_root_body(f"{name}_shaft")
        with elevator_world.modify_world():
            elevator = Elevator.get_annotation_specification(
                name,
                Elevator.get_default_root_kinematic_structure_entity_specification(
                    scale=self.cabin_scale, wall_thickness=self.cabin_wall_thickness
                ),
                parent_connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.Z(),
                    dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                        maximum_speed=self.cabin_maximum_speed
                    ),
                ),
            ).spawn(elevator_world)
            door_scale = Scale(
                self.cabin_door_thickness, self.cabin_scale.y / 2, self.cabin_scale.z
            )
            for i, door_y in enumerate(
                (-self.cabin_scale.y / 4, self.cabin_scale.y / 4)
            ):
                elevator.add(
                    Door.create_with_new_body_in_world(
                        name=f"{name}_door{i}",
                        world=elevator_world,
                        world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            Point3(-self.cabin_scale.x / 2, door_y, 0),
                            reference_frame=elevator_world.root,
                        ),
                        parent_connection_specification=PrismaticConnectionSpecification(
                            axis=(Vector3.Y() * ((-1) ** (i + 1))),
                            dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                                lower_position=0.0, upper_position=door_scale.y
                            ),
                        ),
                        scale=door_scale,
                    )
                )
            elevator_world.add_semantic_annotation(elevator)

        world.merge_world(
            elevator_world,
            FixedConnection(
                parent=world.root,
                child=elevator_world.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.cabin_centre.x,
                    self.cabin_centre.y,
                    self.cabin_scale.z / 2,
                    yaw=np.pi / 2,
                ),
            ),
        )
        world.get_semantic_annotation_by_name(name).drive_to_floor(
            world.get_semantic_annotations_by_type(GroundFloor)[0]
        )

    def create_world(self, root_name: str = "map") -> World:
        """
        :param root_name: Name of the world's root body, the frame the office is in.
        :return: A world holding nothing but this office.
        """
        world = World.create_with_root_body(root_name)
        self.spawn(world)
        return world
