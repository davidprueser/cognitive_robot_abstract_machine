"""
Tiago drives from the lobby of the ARISE office to a given room, riding the elevator
when the room lies on the first floor and driving into the room once the ride is over.

In a kinematic simulation an
:class:`~experiments.tiago_elevator_demo.demo.ElevatorOperator` sends the elevator to the
first floor once Tiago boards it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Iterator

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.exceptions import (
    TargetRoomNotReachedError,
)
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Level,
)
from semantic_digital_twin.world import World

# %% demonstration


@dataclass
class TiagoRoomNavigationDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago drives from the lobby of the ARISE office into a target room.
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
