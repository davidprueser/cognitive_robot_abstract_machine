"""
Take the elevator of the lab elevator room up to the first floor and show the ride in
RViz.

Run with ROS 2 sourced, then in RViz set the fixed frame to ``map`` and add a
MarkerArray display on ``/semworld/viz_marker`` with durability *Transient Local*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.timer import Timer

from experiments.lab_elevator_room_demo import (
    MAP_FRAME,
    RVizWorldPublisher,
    create_lab_elevator_room_world,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    Level,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

logger = logging.getLogger(__name__)

NODE_NAME = "lab_elevator_ride_demo"
"""
Name of the ROS 2 node the demo publishes on.
"""

POSITION_TOLERANCE = 1e-9
"""
Distance below which a joint counts as having reached its target position.
"""

# %% ride


@dataclass
class ElevatorRide:
    """
    Takes an elevator to a level in small steps: it closes the doors, drives the cabin
    to the level, and opens the doors there.
    """

    elevator: Elevator
    """
    The elevator to ride.
    """

    destination: Level
    """
    The level the elevator is taken to.
    """

    speed: float = 0.5
    """
    How fast the doors and the cabin move, in metres per second.
    """

    def __post_init__(self):
        """
        :raises MissingMovableJointError: If the cabin or one of its doors cannot move.
        """
        for part in [self.elevator, *self.elevator.doors]:
            if part.movable_joint is None:
                raise MissingMovableJointError(part)

    @property
    def _door_joints(self) -> list[ActiveConnection1DOF]:
        return [door.movable_joint for door in self.elevator.doors]

    @property
    def _cabin_position(self) -> float:
        """
        :return: The drive position at which the cabin serves :attr:`destination`.
        """
        return self.elevator.drive_position_for_floor(self.destination)

    @property
    def is_finished(self) -> bool:
        """
        :return: Whether the cabin stands at :attr:`destination` with its doors open.
        """
        return self._has_reached(
            self.elevator.movable_joint, self._cabin_position
        ) and all(
            self._has_reached(joint, joint.dof.limits.upper.position)
            for joint in self._door_joints
        )

    def step(self, duration: float) -> None:
        """
        Advance the ride by ``duration`` seconds: close the doors while the cabin is
        away from :attr:`destination`, drive it there once they are closed, and open
        them once it has arrived.

        :param duration: The time the step covers, in seconds.
        """
        distance = self.speed * duration
        cabin = self.elevator.movable_joint
        if self._has_reached(cabin, self._cabin_position):
            for joint in self._door_joints:
                self._move_toward(joint, joint.dof.limits.upper.position, distance)
            return
        if not all(
            self._has_reached(joint, joint.dof.limits.lower.position)
            for joint in self._door_joints
        ):
            for joint in self._door_joints:
                self._move_toward(joint, joint.dof.limits.lower.position, distance)
            return
        self._move_toward(cabin, self._cabin_position, distance)

    @staticmethod
    def _has_reached(joint: ActiveConnection1DOF, target: float) -> bool:
        return abs(joint.position - target) < POSITION_TOLERANCE

    @staticmethod
    def _move_toward(
        joint: ActiveConnection1DOF, target: float, distance: float
    ) -> None:
        """
        Move ``joint`` toward ``target`` by at most ``distance``.
        """
        joint.position = joint.position + float(
            np.clip(target - joint.position, -distance, distance)
        )


# %% demo


@dataclass
class LabElevatorRideDemo:
    """
    Publishes the lab elevator room to RViz and, after a short delay, rides its elevator
    from the ground floor up to the first floor.
    """

    node: Node
    """
    The node the world is published on and the ride is timed by.
    """

    start_delay_seconds: float = 3.0
    """
    How long the elevator waits on the ground floor before the ride starts, so that RViz
    shows its starting position.
    """

    step_period_seconds: float = 0.05
    """
    How often the ride is advanced.
    """

    world: World = field(init=False)
    """
    The world holding the lab elevator room.
    """

    publisher: RVizWorldPublisher = field(init=False)
    """
    Publishes :attr:`world` to RViz.
    """

    ride: ElevatorRide = field(init=False)
    """
    The ride of the room's elevator to the first floor.
    """

    _start_timer: Timer = field(init=False)
    """
    Fires once the start delay has passed.
    """

    _step_timer: Timer | None = field(init=False, default=None)
    """
    Advances the ride while it is under way.
    """

    def __post_init__(self):
        self.world = create_lab_elevator_room_world()
        self.publisher = RVizWorldPublisher(world=self.world, node=self.node)
        (elevator,) = self.world.get_semantic_annotations_by_type(Elevator)
        (first_floor,) = self.world.get_semantic_annotations_by_type(FirstFloor)
        self.ride = ElevatorRide(elevator=elevator, destination=first_floor)
        self._start_timer = self.node.create_timer(
            self.start_delay_seconds, self._start
        )

    def _start(self) -> None:
        self._start_timer.cancel()
        logger.info("Taking the elevator up to the first floor.")
        self._step_timer = self.node.create_timer(self.step_period_seconds, self._step)

    def _step(self) -> None:
        self.ride.step(self.step_period_seconds)
        if not self.ride.is_finished:
            return
        self._step_timer.cancel()
        logger.info("The elevator has arrived at the first floor.")


def main() -> None:
    """
    Publish the lab elevator room to RViz and ride its elevator to the first floor, then
    keep publishing until interrupted.
    """
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    LabElevatorRideDemo(node=node)
    logger.info(
        "Publishing the lab elevator room. In RViz set the fixed frame to '%s' and add "
        "a MarkerArray display on '%s' with durability 'Transient Local'.",
        MAP_FRAME,
        VizMarkerPublisher.topic_name,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
