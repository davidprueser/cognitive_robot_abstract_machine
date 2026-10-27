"""
Tiago waits in the waiting zone in front of the open elevator of a building, the lab by
default, senses the people around it through ROS4HRI and hands its base over to the
social navigation module (IRII), which drives it around the people into the cabin.

The people come from a ROS4HRI person manager: on the real TIAGo PAL's, in the Gazebo
simulator ``tiago_arise_humans`` (scenario ``elevator_waiting``). Only the persons
inside the cabin and in the waiting area in front of its doors on the robot's level are
relevant to boarding and go into the request.

While the social navigation module has the base, the plan sends no motions; the module
drives the robot through ``/cmd_vel`` and gives the base back with its result. Request
and result are JSON on :class:`SocialNavigationTopic`, a stand-in until the interface with
IRII is agreed; ``mock_social_navigation`` in ``tiago_arise_humans`` plays the module in
simulation.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from rclpy.duration import Duration
from std_msgs.msg import String
from typing_extensions import Any

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from experiments.ra_fetch_elevator_people.elevator_people import ElevatorPeople
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_elevator_waiting_demo.exceptions import (
    BoardingFailedError,
    BoardingTimedOutError,
)
from semantic_digital_twin.reasoning.elevator_areas import (
    ElevatorArea,
    ElevatorCabinArea,
    ElevatorWaitingArea,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator
from semantic_digital_twin.spatial_types import Point3, Pose
from semantic_digital_twin.world import World

# %% the request to the social navigation module


class SocialNavigationTopic(StrEnum):
    """
    Topics of the mock interface to the social navigation module, each carrying JSON in
    a ``std_msgs/String``.
    """

    BOARDING_REQUEST = "/ra_fetch/social_navigation/boarding_request"
    BOARDING_RESULT = "/ra_fetch/social_navigation/boarding_result"


class ElevatorAreaName(StrEnum):
    """
    The areas around an elevator a person can be in.
    """

    CABIN = "cabin"
    WAITING_AREA = "waiting_area"


@dataclass
class PersonNearElevator:
    """
    A person relevant to boarding the elevator, as sent to the social navigation module.
    """

    person_id: str
    """
    The person's ROS4HRI person ID.
    """

    area: ElevatorAreaName
    """
    Where around the elevator the person is.
    """

    position: list[float]
    """
    The person's position, at head height, in the request's frame.
    """


@dataclass
class ElevatorDescription:
    """
    Where the elevator's doors and cabin are, as the social navigation module needs them
    to drive the robot in. Positions lie on the cabin's floor.
    """

    door_center: list[float]
    """
    The middle of the doorway.
    """

    cabin_center: list[float]
    """
    The middle of the cabin.
    """

    inward_yaw: float
    """
    The direction from the doors into the cabin, in radians.
    """

    cabin_width: float
    """
    The cabin's inner width, along the doors.
    """

    cabin_depth: float
    """
    The cabin's inner depth, from the doors to the back wall.
    """

    @classmethod
    def of(cls, elevator: Elevator) -> ElevatorDescription:
        """
        :return: The description of ``elevator`` in its world's root frame.
        """
        world = elevator._world
        scale = elevator.scale
        door_center, cabin_center = (
            world.transform(
                Point3(x, 0.0, -scale.z / 2, reference_frame=elevator.root), world.root
            )
            for x in (-scale.x / 2, 0.0)
        )
        return cls(
            door_center=door_center.to_np()[:3].tolist(),
            cabin_center=cabin_center.to_np()[:3].tolist(),
            inward_yaw=float(elevator.root.global_transform.rotation_matrix.rpy[2]),
            cabin_width=float(scale.y),
            cabin_depth=float(scale.x),
        )


@dataclass
class BoardingRequest:
    """
    Hands the robot's base over to the social navigation module, so that it drives the
    robot into the elevator's cabin.
    """

    request_id: int
    """
    Number the result refers back to.
    """

    frame_id: str
    """
    The frame all positions are given in.
    """

    robot_position: list[float]
    """
    Position of the robot's base.
    """

    robot_yaw: float
    """
    Heading of the robot's base in radians.
    """

    elevator: ElevatorDescription
    """
    The elevator to board.
    """

    persons: list[PersonNearElevator]
    """
    The persons inside the cabin and in the waiting area in front of it.
    """

    def to_message(self) -> String:
        """
        :return: This request as the JSON the social navigation module reads.
        """
        return String(data=json.dumps(asdict(self)))


class BoardingStatus(StrEnum):
    """
    How boarding ended when the social navigation module gave the base back.
    """

    SUCCEEDED = "SUCCEEDED"
    """
    The robot stands inside the cabin.
    """

    FAILED = "FAILED"
    """
    The robot could not board; it stands wherever the module stopped it.
    """


@dataclass
class BoardingResult:
    """
    The social navigation module's answer to a :class:`BoardingRequest`, which gives the
    base back to the plan.
    """

    request_id: int
    """
    The request this result answers.
    """

    status: BoardingStatus
    """
    Whether the robot boarded.
    """

    reason: str
    """
    Why, in words, for the log.
    """

    @classmethod
    def from_json(cls, content: dict[str, Any]) -> BoardingResult:
        """
        :param content: The decoded JSON the social navigation module sent.
        :return: The result it describes.
        """
        return cls(
            request_id=content["request_id"],
            status=BoardingStatus(content["status"]),
            reason=content["reason"],
        )


# %% demonstration


@dataclass
class TiagoElevatorWaitingDemonstration(TiagoElevatorRideDemonstration):
    """
    Tiago drives to the waiting zone in front of the elevator, senses who is inside and
    in front of it, and hands its base over to the social navigation module to board.
    """

    ros_node_name: str = "tiago_elevator_waiting_demo_node"

    distance_from_doors: float = 1.5
    """
    How far in front of the elevator's doors the robot waits.
    """

    sensing_time: float = 3.0
    """
    Seconds the robot watches the tracked persons before it sends the request.
    """

    sensing_period: float = 0.2
    """
    Seconds between two looks at the tracked persons while sensing.
    """

    request_id: int = 1
    """
    Number of the boarding request this demonstration sends.
    """

    queue_depth: int = 10
    """
    How many messages the boarding request publisher and result subscriber hold.
    """

    boarding_timeout: float = 120.0
    """
    Seconds the social navigation module may keep the base before the plan gives up
    waiting for its result.
    """

    result: BoardingResult | None = field(init=False, default=None)
    """
    The result the social navigation module sent back, once it arrived.
    """

    def waiting_pose(self, elevator: Elevator) -> Pose:
        """
        :return: Where the robot waits: :attr:`distance_from_doors` in front of the
            middle of the doors, facing the cabin, on the cabin's floor.
        """
        return Pose.from_xyz_rpy(
            x=-elevator.scale.x / 2 - self.distance_from_doors,
            z=-elevator.scale.z / 2,
            reference_frame=elevator.root,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Park the arms and drive to the waiting zone.
        """
        elevator = context.world.get_semantic_annotations_by_type(Elevator)[0]
        return sequential(
            [
                ParkArmsAction(context.robot.all_arms),
                NavigateAction(self.waiting_pose(elevator)),
            ],
            context=context,
        )

    def conclude(self, world: World) -> None:
        """
        Sense the relevant persons, hand the base over to the social navigation module
        with the boarding request and wait until it gives the base back.

        :raises BoardingTimedOutError: If no result arrived within
            :attr:`boarding_timeout`.
        :raises BoardingFailedError: If the robot did not board.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        elevator = world.get_semantic_annotations_by_type(Elevator)[0]
        node = self.ros_node
        elevator_people = ElevatorPeople.for_world(
            node, world, robot.root.name.name, None
        )
        result_arrived = threading.Event()

        def on_result(message: String) -> None:
            result = BoardingResult.from_json(json.loads(message.data))
            if result.request_id != self.request_id:
                return
            self.result = result
            result_arrived.set()

        node.create_subscription(
            String, SocialNavigationTopic.BOARDING_RESULT, on_result, self.queue_depth
        )
        publisher = node.create_publisher(
            String, SocialNavigationTopic.BOARDING_REQUEST, self.queue_depth
        )
        sensing_end = node.get_clock().now() + Duration(seconds=self.sensing_time)
        while node.get_clock().now() < sensing_end:
            elevator_people.update()
            time.sleep(self.sensing_period)

        request = self._boarding_request(elevator_people, elevator, robot)
        node.get_logger().info(
            f"Handing the base over to the social navigation module: "
            f"{request.to_message().data}"
        )
        publisher.publish(request.to_message())
        result_arrived.wait(self.boarding_timeout)
        elevator_people.mirror.keep_only([])
        if self.result is None:
            raise BoardingTimedOutError(timeout=self.boarding_timeout)
        node.get_logger().info(f"Base is back, boarding result: {self.result}")
        if self.result.status is not BoardingStatus.SUCCEEDED:
            raise BoardingFailedError(result=self.result)

    def _boarding_request(
        self, elevator_people: ElevatorPeople, elevator: Elevator, robot: Tiago
    ) -> BoardingRequest:
        """
        :return: The request to board ``elevator``, with the robot's pose and the
            persons ``elevator_people`` mirrored inside the cabin and in the waiting
            area on the robot's level.
        """
        areas: dict[ElevatorAreaName, ElevatorArea] = {
            ElevatorAreaName.CABIN: ElevatorCabinArea(elevator)
        }
        robot_level = elevator_people.robot_level()
        if robot_level is not None:
            areas[ElevatorAreaName.WAITING_AREA] = ElevatorWaitingArea(
                elevator, robot_level
            )
        mirror = elevator_people.mirror
        persons = [
            PersonNearElevator(
                person_id=person_id,
                area=area_name,
                position=mirror.humans[person_id]
                .root.global_transform.to_np()[:3, 3]
                .tolist(),
            )
            for area_name, area in areas.items()
            for person_id in mirror.persons_in(area)
        ]
        world_T_robot = robot.root.global_transform
        return BoardingRequest(
            request_id=self.request_id,
            frame_id=elevator._world.root.name.name,
            robot_position=world_T_robot.to_np()[:3, 3].tolist(),
            robot_yaw=float(world_T_robot.rotation_matrix.rpy[2]),
            elevator=ElevatorDescription.of(elevator),
            persons=persons,
        )


def main(execution_type: ExecutionType = ExecutionType.REAL) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real (or Gazebo) robot or simulate it.
    """
    TiagoElevatorWaitingDemonstration(
        used_robot=Tiago, execution_type=execution_type
    ).run()


if __name__ == "__main__":
    main()
