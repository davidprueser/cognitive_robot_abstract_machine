from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import Optional, Any, Dict

from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import NoFloorBelowRobot, NotOnASingleLevelException
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.factories import execute_single, pause_until, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.speech import SayAction
from coraplex.robot_plans.motions.navigation import MoveMotion, TurnMotion
from coraplex.robot_plans.motions.robot_body import LookingMotion
from giskardpy.motion_statechart.monitors.joint_monitors import (
    JointPositionReached,
)
from giskardpy.motion_statechart.monitors.sensor_monitors import PassageClear
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable_from, and_, ConditionType
from semantic_digital_twin.exceptions import MissingMovableJointError
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.robots.exceptions import MissingLidarError
from semantic_digital_twin.robots.robot_part_mixins import HasLidar
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.semantic_annotations.mixins import HasPlanarFreeSpace
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Level,
    Elevator,
    Floor,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    HomogeneousTransformationMatrix,
    Point2,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Region


@dataclass
class NavigateAction(ActionDescription):
    """
    Navigates the Robot to a position.
    """

    target_location: Pose
    """
    Where the robot should stand, and which way it should face given as the pose's
    x-axis.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveMotion(self.robot.mobile_base.pose_facing(self.target_location))
        )

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The robot needs to have a drive and the target location needs to be free from
        obstacles.
        """
        drive_variable = variable_from(context.robot.drive is not None)
        return and_(
            is_pose_free_for_robot(context.robot, variables["target_location"]),
            drive_variable,
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> ConditionType:
        """
        The robot needs to be within 3 cm of where the heading puts its base.
        """
        return allclose(
            variable_from(context.robot.root).global_pose,
            context.robot.mobile_base.pose_facing(kwargs["target_location"]),
            atol=0.03,
        )


@dataclass
class LookAtAction(ActionDescription):
    """
    Lets the robot look at a position.
    """

    target: Pose
    """
    Position at which the robot should look, given as 6D pose.
    """

    camera: Optional[Camera] = None
    """
    Camera that should be looking at the target.
    """

    @property
    def _action_plan(self) -> PlanNode:
        camera = self.camera or self.robot.get_default_camera()
        return execute_single(LookingMotion(target=self.target, camera=camera))


@dataclass
class FaceAtAction(ActionDescription):
    """
    Turns the robot's base on the spot until its front faces a target.
    """

    target: Pose
    """
    What to face; only its horizontal position matters.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(TurnMotion(self.target))


@dataclass
class PathPlanningNavigateAction(ActionDescription):
    """
    Navigates the robot to a pose along a path through the environment's free space.

    The free space is decomposed into a graph of convex sets, so the robot drives around
    the furniture and walls between it and the target instead of straight at them. It is
    the free space of the level the robot is on, or, outside every level, of the floor
    it stands on.

    This works for obstacles which are known in the environment beforehand, not for
    those added during navigation.
    """

    target: Pose
    """
    Where the robot should stand at the end of the path, with its base.
    """

    obstacle_clearance: float = field(default=0.0, kw_only=True)
    """
    How far the robot's base keeps from obstacles on top of its base radius.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([MoveMotion(waypoint) for waypoint in self._path()])

    @property
    def _ground(self) -> HasPlanarFreeSpace:
        """
        Where the path is laid out: the level whose region alone holds the robot, or
        else the floor it stands on.

        :raises NoFloorBelowRobot: If the robot is on no single level and stands over no
            annotated floor.
        """
        levels = Level.holding(self.robot.bodies_with_collision[0])
        if len(levels) == 1:
            return levels[0]
        return self._floor

    @property
    def _floor(self) -> Floor:
        """
        The floor the robot stands on.

        :raises NoFloorBelowRobot: If the robot stands over no annotated floor.
        """
        floor = Floor.under(self.robot.root)
        if floor is None:
            raise NoFloorBelowRobot(self.robot)
        return floor

    def _path(self) -> list[Pose]:
        """
        The poses the robot drives to, one per leg of the path.

        Each pose faces the waypoint after it, so the leg leaving a waypoint no longer
        has to begin by turning. The waypoint the robot already stands on is left out,
        and the last pose is the requested target.

        .. note::
            The orientation aims the base's x-axis, which is the axis a drive travels
            along, rather than the base's
            :attr:`~semantic_digital_twin.robots.robot_parts.MobileBase.forward_axis`.
            The two differ on a base whose front is not its direction of travel, and it
            is travel that these orientations exist to line up.

        :return: The poses to drive to, in order.
        """
        waypoints = self._waypoints()
        base_height = self.world.transform(
            self.robot.root.global_transform, waypoints[0].reference_frame
        ).z
        poses = [
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(waypoint.x, waypoint.y, base_height, waypoint.reference_frame),
                RotationMatrix.from_vectors(
                    x=Vector3(
                        next_waypoint.x - waypoint.x,
                        next_waypoint.y - waypoint.y,
                        0,
                        reference_frame=waypoint.reference_frame,
                    ),
                    z=Vector3.Z(),
                    reference_frame=waypoint.reference_frame,
                ),
                reference_frame=waypoint.reference_frame,
            ).pose
            for waypoint, next_waypoint in zip(waypoints[1:], waypoints[2:])
        ]
        return poses + [self.target]

    def _waypoints(self) -> list[Point2]:
        """
        The points the robot travels through to get from where it stands to the target.

        :return: The path, beginning at the robot's own position and ending at the
            target's.
        """
        base_pose = self.robot.root.global_pose
        free_space = self._ground.planar_free_space(
            max_height=self.robot.as_bounding_box_collection_in_frame(self.robot.root)
            .bounding_box()
            .scale.z,
            bloat_obstacles=self.robot.mobile_base.base_radius
            + self.obstacle_clearance,
        )
        return free_space.path_from_to(
            Point2.from_pose(base_pose), Point2.from_pose(self.target)
        )


@dataclass
class PassDoorAction(ActionDescription):
    """
    Drives the robot through a doorway towards a destination: it lines up in front of
    the doorway on the side away from the destination, waits until its lidar sees a
    passage as wide as its base, and drives through to the destination's side.
    """

    doors: list[Door]
    """
    The doors covering the doorway, such as both leaves of a double door.
    """

    destination: Point3
    """
    A point on the side of the doorway the robot should end up on.
    """

    approach_distance: float = field(default=0.5, kw_only=True)
    """
    Distance from the doorway at which the robot waits for the passage to clear.
    """

    exit_distance: float = field(default=0.5, kw_only=True)
    """
    Distance from the doorway at which the robot stops on the other side.
    """

    waiting_announcement: str | None = field(default=None, kw_only=True)
    """
    What the robot says once it stands in front of the doorway, before it waits for the
    passage to clear, or None to wait silently.
    """

    @property
    def _action_plan(self) -> PlanNode:
        side = self._destination_side
        steps = [NavigateAction(self._pose_at(-side * self.approach_distance, side))]
        if self.waiting_announcement is not None:
            steps.append(SayAction(self.waiting_announcement))
        steps.append(
            pause_until(
                [NavigateAction(self._pose_at(side * self.exit_distance, side))],
                monitor=self._passage_clear(),
            )
        )
        return sequential(steps)

    @property
    def _doorway_frame(self) -> Region:
        """
        The frame the doorway is measured in: the entry way of the first door, whose
        x-axis points through the doorway.
        """
        return self.doors[0].entry_way.root

    @property
    def _doorway(self) -> VolumetricBoundingBox:
        """
        The space all entry ways of the doors span together, in :attr:`_doorway_frame`.
        """
        return BoundingBoxCollection(
            shapes=[
                box
                for door in self.doors
                for box in door.entry_way.root.area.as_bounding_box_collection_in_frame(
                    self._doorway_frame
                )
            ],
            reference_frame=self._doorway_frame,
        ).bounding_box()

    @property
    def _destination_side(self) -> int:
        """
        ``1`` if :attr:`destination` lies on the positive x side of the doorway, ``-1``
        otherwise.
        """
        destination_x = self.world.transform(self.destination, self._doorway_frame).x
        return 1 if float(destination_x) >= float(self._doorway.center.x) else -1

    @property
    def _height_in_doorway_frame(self) -> float:
        """
        The robot's height in :attr:`_doorway_frame`, which its drive cannot change.
        """
        return float(
            self.world.transform(
                self.robot.root.global_transform, self._doorway_frame
            ).z
        )

    def _pose_at(self, distance: float, heading: int) -> Pose:
        """
        :param distance: Signed distance from the doorway's centre along the doorway
            frame's x-axis.
        :param heading: ``1`` to face along the doorway frame's x-axis, ``-1`` against
            it.
        :return: A pose centred on the doorway at that distance.
        """
        return Pose.from_xyz_rpy(
            x=float(self._doorway.center.x) + distance,
            y=float(self._doorway.center.y),
            z=self._height_in_doorway_frame,
            yaw=0.0 if heading > 0 else np.pi,
            reference_frame=self._doorway_frame,
        )

    def _passage_clear(self) -> PassageClear:
        """
        Observes True from the first time the robot's lidar sees a passage as wide as
        its base through the centre of the doorway.

        :raises MissingLidarError: If the robot's mobile base carries no lidar.
        """
        mobile_base = self.robot.mobile_base
        if not isinstance(mobile_base, HasLidar):
            raise MissingLidarError(robot_part=mobile_base)
        passage_clear = PassageClear(
            lidar=mobile_base.lidar,
            passage_start=Point3(
                self._doorway.center.x,
                self._doorway.center.y - mobile_base.base_radius,
                self._height_in_doorway_frame,
                reference_frame=self._doorway_frame,
            ),
            passage_end=Point3(
                self._doorway.center.x,
                self._doorway.center.y + mobile_base.base_radius,
                self._height_in_doorway_frame,
                reference_frame=self._doorway_frame,
            ),
            name="DoorwayClear",
            stays_clear_once_seen=True,
        )
        return passage_clear


class ElevatorRideAnnouncement(StrEnum):
    """
    What a robot riding an elevator says about where it believes it is and where it is
    headed, with the floors left open.
    """

    DEPARTING = "I am on the {current_floor} and heading to the {target_floor}."
    """
    Said before the robot sets off towards the elevator.
    """

    WAITING_AT_DOORS = (
        "I am in front of the elevator and waiting for it on the {current_floor}."
    )
    """
    Said once the robot stands in front of the elevator's doors.
    """

    RIDING = "I am inside the elevator and riding to the {target_floor}."
    """
    Said once the robot has boarded the cabin.
    """

    ARRIVED = "I have arrived on the {target_floor}."
    """
    Said once the robot has left the cabin on the floor it was headed to.
    """

    def verbalize(self, current_floor: Level, target_floor: Level) -> str:
        """
        :param current_floor: The floor the robot sets off from.
        :param target_floor: The floor the robot is headed to.
        :return: This announcement with the floors filled in by their spoken names.
        """
        return self.format(
            current_floor=current_floor.spoken_name,
            target_floor=target_floor.spoken_name,
        )


@dataclass
class ElevatorNavigation(ActionDescription):
    """
    Navigates a robot to another level of a building using an elevator, the robot drives
    in the elevator, turns towards the doors and waits there until its lidar sees the
    doors open again and the elevator is at the right level.

    Along the way the robot says where it believes it is and where it is headed, see
    :class:`ElevatorRideAnnouncement`.
    """

    elevator: Elevator
    """
    Elevator the robot rides.
    """

    target_floor: Level
    """
    Level of the building the robot should end up on.
    """

    exit_clearance: float = field(default=0.5, kw_only=True)
    """
    Distance the robot keeps from the elevator's opening after driving out, on top of
    half the cabin's depth.
    """

    arrival_threshold: float = field(default=0.01, kw_only=True)
    """
    Position error within which the elevator's drive counts as having arrived.
    """

    @property
    def _action_plan(self) -> PlanNode:
        half_cabin_depth = float(self.elevator.scale.x) / 2
        current_floor = self._current_floor
        return sequential(
            [
                SayAction(
                    ElevatorRideAnnouncement.DEPARTING.verbalize(
                        current_floor, self.target_floor
                    )
                ),
                pause_until(
                    [
                        PassDoorAction(
                            self.elevator.doors,
                            destination=Point3(reference_frame=self.elevator.root),
                            approach_distance=self.exit_clearance,
                            exit_distance=half_cabin_depth,
                            waiting_announcement=ElevatorRideAnnouncement.WAITING_AT_DOORS.verbalize(
                                current_floor, self.target_floor
                            ),
                        )
                    ],
                    monitor=self._cabin_at_floor(current_floor),
                ),
                ReAttachNode(body=self.robot.root, new_parent=self.elevator.root),
                SayAction(
                    ElevatorRideAnnouncement.RIDING.verbalize(
                        current_floor, self.target_floor
                    )
                ),
                pause_until(
                    [
                        PassDoorAction(
                            self.elevator.doors,
                            destination=self._point_outside_cabin,
                            approach_distance=half_cabin_depth,
                            exit_distance=self.exit_clearance,
                        )
                    ],
                    monitor=self._cabin_at_floor(self.target_floor),
                ),
                ReAttachNode(body=self.robot.root, new_parent=self.world.root),
                SayAction(
                    ElevatorRideAnnouncement.ARRIVED.verbalize(
                        current_floor, self.target_floor
                    )
                ),
            ]
        )

    @property
    def _current_floor(self) -> Level:
        """
        Finds the floor the robot is currently on, based on its position in the world.

        Raises :class:`WrongLevelException` if the robot is not on any floor or on
        multiple floors at once.
        :return: The semantic annotation for the floor
        """
        current_floor = Level.holding(self.robot.bodies_with_collision[0])
        if len(current_floor) == 0:
            raise NotOnASingleLevelException("Robot is not on any recognized floor.")
        if len(current_floor) > 1:
            raise NotOnASingleLevelException("Robot is on multiple floors at once.")
        return current_floor[0]

    @property
    def _point_outside_cabin(self) -> Point3:
        """
        A point a cabin depth beyond the elevator's opening.
        """
        cabin_depth = float(self.elevator.scale.x)
        hole_direction = self.elevator.hole_direction
        return Point3(
            float(hole_direction[0]) * cabin_depth,
            float(hole_direction[1]) * cabin_depth,
            float(hole_direction[2]) * cabin_depth,
            reference_frame=self.elevator.root,
        )

    def _cabin_at_floor(self, floor: Level) -> JointPositionReached:
        """
        Observes True once the cabin serves ``floor``.

        :raises MissingMovableJointError: If the elevator has no drive.
        """
        if self.elevator.movable_joint is None:
            raise MissingMovableJointError(self.elevator)
        return JointPositionReached(
            connection=self.elevator.movable_joint,
            position=self.elevator.drive_position_for_floor(floor),
            threshold=self.arrival_threshold,
            name="ElevatorAtFloor",
        )
