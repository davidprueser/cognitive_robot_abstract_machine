"""
Driving a robot from the room it is in towards another room of its building, riding the
elevator when the other room lies on another level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    PathPlanningNavigateAction,
)
from experiments.tiago_room_navigation_demo.exceptions import (
    NoElevatorError,
    RoomOnNoLevelError,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Level,
    Room,
)

# %% navigating to a room


@dataclass
class NavigateToRoomAction(ActionDescription):
    """
    Drives the robot one leg of the way to a room, along a path through the free space
    of the level it stands on.

    A room on the robot's own level is driven to directly, to its middle. For a room on
    another level the robot drives in front of the elevator's doors and rides the
    elevator to the room's level; a second leg then drives into the room.

    .. note:: A leg is laid out from where the robot stands when it is planned, so the
        leg after a ride can only be planned once the ride is over.

    .. note:: The robot rides the first elevator of its world.
    """

    target_room: Room
    """
    The room the robot should end up in.
    """

    elevator_waiting_distance: float = field(default=1.5, kw_only=True)
    """
    How far in front of the elevator's doors the robot drives before riding it.
    """

    elevator_arrival_threshold: float = field(default=0.01, kw_only=True)
    """
    Position error within which the elevator's drive counts as having arrived.
    """

    obstacle_clearance: float = field(default=0.0, kw_only=True)
    """
    How far the robot's base keeps from obstacles on top of its base radius while it
    drives.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        :raises RoomOnNoLevelError: If the target room lies on no level.
        :raises NoElevatorError: If the target room lies on another level than the
            robot and the world holds no elevator.
        """
        target_level = Level.under(self.target_room.floor.root)
        if target_level is None:
            raise RoomOnNoLevelError(self.target_room.name.name)
        if Level.under(self.robot.root) is target_level:
            return sequential(
                [
                    PathPlanningNavigateAction(
                        self.target_room.floor_centre,
                        obstacle_clearance=self.obstacle_clearance,
                    )
                ]
            )
        elevators = self.world.get_semantic_annotations_by_type(Elevator)
        if not elevators:
            raise NoElevatorError(self.target_room.name.name)
        elevator = elevators[0]
        return sequential(
            [
                PathPlanningNavigateAction(
                    elevator.pose_in_front_of_doors(self.elevator_waiting_distance),
                    obstacle_clearance=self.obstacle_clearance,
                ),
                ElevatorNavigation(
                    elevator,
                    target_level,
                    arrival_threshold=self.elevator_arrival_threshold,
                ),
            ]
        )
