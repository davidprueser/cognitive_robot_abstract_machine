"""
Tiago drives from a start room of an office to a target room, riding the elevator when
the target room lies on another level and driving into the room once the ride is over.

In a kinematic simulation an
:class:`~experiments.tiago_elevator_demo.demo.ElevatorOperator` sends the elevator to the
target room's level once Tiago boards it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Iterator

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from experiments.tiago_arise_gazebo.entity_teleport import SimulatedEntityTeleport
from experiments.tiago_arise_gazebo.services import SimulatorService
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.exceptions import (
    RobotNotPlacedError,
    RoomOnNoLevelError,
    TargetRoomNotReachedError,
)
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.iai_office import (
    IAIOffice,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Level,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoRoomNavigationDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago drives from the start room into the target room.
    """

    ros_node_name: str = "tiago_room_navigation_demo_node"

    target_room_name: str = field(kw_only=True)
    """
    The name of the room annotation Tiago drives to.
    """

    obstacle_clearance: float = field(default=0.0, kw_only=True)
    """
    How far Tiago's base keeps from obstacles on top of its base radius while it drives.
    """

    elevator_exit_clearance: float = field(default=0.5, kw_only=True)
    """
    Distance Tiago keeps from the elevator's opening after driving out of the cabin, on
    top of half the cabin's depth.
    """

    entity_teleport_service: str | None = field(
        default=SimulatorService.SET_ENTITY_POSE, kw_only=True
    )
    """
    The service a real run moves the simulated Tiago into the start room through, or
    None where Tiago is a real robot that starts wherever it stands.
    """

    simulated_robot_name: str = field(default="tiago", kw_only=True)
    """
    The name of Tiago's model in the simulated world.
    """

    placement_timeout: float = field(default=30.0, kw_only=True)
    """
    Seconds a real run waits for the world to show Tiago in the start room.
    """

    placement_tolerance: float = field(default=0.05, kw_only=True)
    """
    How far Tiago may stand from :meth:`start_pose` and still count as placed there.
    """

    def perform(self, world: World) -> None:
        """
        Put Tiago in the start room first, see :meth:`place_robot_in_start_room`.
        """
        self.place_robot_in_start_room(world)
        super().perform(world)

    def place_robot_in_start_room(self, world: World) -> None:
        """
        Stand Tiago on the map at :meth:`start_pose`.

        A real run moves the simulated Tiago there and waits for the world to show it
        there; a real Tiago is left where it stands.

        :raises RobotNotPlacedError: If a real run's world does not show Tiago in the
            start room within :attr:`placement_timeout`.
        """
        if (
            self.execution_type is ExecutionType.REAL
            and self.entity_teleport_service is None
        ):
            return
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        start = self.start_pose(world)
        self._stand_robot_on_map(robot, float(start.z))
        if self.execution_type is not ExecutionType.REAL:
            robot.root.parent_connection.origin = start.homogeneous_matrix
            return
        SimulatedEntityTeleport(
            node=self.ros_node, service_name=self.entity_teleport_service
        ).teleport(self.simulated_robot_name, start)
        self._wait_until_robot_stands_at(robot, start)

    def _stand_robot_on_map(self, robot: AbstractRobot, height: float) -> None:
        """
        Hang ``robot``'s drive off the map at ``height``, keeping where on the map it
        stands.
        """
        world = robot._world
        world.move_branch(robot.root, world.root)
        lift = height - float(robot.root.global_transform.z)
        if np.isclose(lift, 0.0):
            return
        drive = robot.root.parent_connection
        lifted_drive = drive.copy_with_new_parent(
            world.root,
            HomogeneousTransformationMatrix(
                (
                    HomogeneousTransformationMatrix.from_xyz_rpy(z=lift)
                    @ drive.parent_T_connection_expression
                ).evaluate()
            ),
        )
        with world.modify_world():
            world.remove_connection(drive)
            world.add_connection(lifted_drive)

    def _wait_until_robot_stands_at(self, robot: AbstractRobot, pose: Pose) -> None:
        """
        :raises RobotNotPlacedError: If ``robot`` does not stand within
            :attr:`placement_tolerance` of ``pose`` within :attr:`placement_timeout`.
        """
        deadline = time.monotonic() + self.placement_timeout
        while (
            np.linalg.norm(
                robot.root.global_transform.position.to_np()[:2]
                - pose.position.to_np()[:2]
            )
            > self.placement_tolerance
        ):
            if time.monotonic() > deadline:
                raise RobotNotPlacedError(self.start_room_name)
            time.sleep(0.1)

    def destination_level(self, world: World) -> Level:
        """
        :return: The level of ``world`` the target room lies on.
        :raises RoomOnNoLevelError: If the target room lies on no level.
        """
        target_room = world.get_semantic_annotation_by_name(self.target_room_name)
        level = Level.under(target_room.floor.root)
        if level is None:
            raise RoomOnNoLevelError(self.target_room_name)
        return level

    def build_plan(self, context: Context) -> PlanNode:
        """
        Drive one leg of the way to the target room.
        """
        return sequential(
            [
                NavigateToRoomAction(
                    context.world.get_semantic_annotation_by_name(
                        self.target_room_name
                    ),
                    elevator_arrival_threshold=self.elevator_arrival_threshold,
                    obstacle_clearance=self.obstacle_clearance,
                    elevator_exit_clearance=self.elevator_exit_clearance,
                )
            ],
            context=context,
        )

    def build_plans(self, context: Context) -> Iterator[PlanNode]:
        """
        Ride to the target room's level first when Tiago stands on another one, then
        drive into the room.
        """
        target_room = context.world.get_semantic_annotation_by_name(
            self.target_room_name
        )
        if Level.under(context.robot.root) is not Level.under(target_room.floor.root):
            yield self.build_plan(context)
        yield self.build_plan(context)

    def conclude(self, world: World) -> None:
        """
        :raises TargetRoomNotReachedError: If Tiago does not stand on the target room's
            floor.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        target_room = world.get_semantic_annotation_by_name(self.target_room_name)
        if Floor.under(robot.root) is not target_room.floor:
            raise TargetRoomNotReachedError(self.target_room_name)


# %% the IAI's office


@dataclass
class TiagoIAIOfficeRoomNavigationDemonstration(TiagoRoomNavigationDemonstration):
    """
    Tiago drives from a start room into a target room of the IAI's office.
    """

    office: IAIOffice = field(default_factory=IAIOffice)

    building_name: str = "lab"

    start_room_name: str = field(default="lab_ground_floor_room_0.31", kw_only=True)

    target_room_name: str = field(default="lab_first_floor_room_1.54", kw_only=True)

    obstacle_clearance: float = field(default=0.15, kw_only=True)

    elevator_exit_clearance: float = field(default=0.7, kw_only=True)
