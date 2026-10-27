"""
Who is inside the elevator and who waits in front of it, from the people a ROS4HRI
(REP-155) pipeline tracks.

Tracked persons are mirrored into the semantic digital twin as :class:`Human` bodies at
their ``person_<personID>`` TF frame. Each update publishes, as ``hri_msgs/IdsList`` of
person IDs, who is inside the cabin and who is in the waiting area in front of its doors
on the level the robot is on, along with both areas' outlines.

Running :func:`main` against the TIAGo ARISE Gazebo simulator with its
``tiago_arise_humans`` launch gives the real data flow without a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import rclpy
from geometry_msgs.msg import Point32, PolygonStamped
from hri_msgs.msg import IdsList
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.time import Time
from sensor_msgs.msg import JointState
from typing_extensions import Self

from semantic_digital_twin.adapters.ros.latest_message_subscriber import (
    LatestMessageSubscriber,
)
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.lab_building import (
    LabBuilding,
)
from semantic_digital_twin.reasoning.elevator_areas import (
    ElevatorArea,
    ElevatorCabinArea,
    ElevatorWaitingArea,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    GroundFloor,
    Human,
    Level,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% topics


class Ros4HriTopic(StrEnum):
    """
    Topics of REP-155 this module reads.
    """

    TRACKED_PERSONS = "/humans/persons/tracked"


class SimulatorTopic(StrEnum):
    """
    Topics of the TIAGo ARISE Gazebo simulator this module reads.
    """

    ELEVATOR_JOINT_STATES = "/elevator/joint_states"


class RobotFrame(StrEnum):
    """
    TF frames of the robot this module looks up.
    """

    BASE = "base_footprint"


class ElevatorPeopleTopic(StrEnum):
    """
    Topics this module publishes.
    """

    CABIN_PERSONS = "/ra_fetch/elevator/cabin/persons"
    CABIN_FOOTPRINT = "/ra_fetch/elevator/cabin/footprint"
    WAITING_AREA_PERSONS = "/ra_fetch/elevator/waiting_area/persons"
    WAITING_AREA_FOOTPRINT = "/ra_fetch/elevator/waiting_area/footprint"


# %% tracked persons


@dataclass
class TrackedPersons(LatestMessageSubscriber[IdsList]):
    """
    The persons a ROS4HRI person manager currently tracks.
    """

    @property
    def person_ids(self) -> list[str]:
        """
        :return: The IDs of the tracked persons, none before the first message.
        """
        if self.latest_message is None:
            return []
        return list(self.latest_message.ids)


@dataclass
class PersonMirror:
    """
    Keeps one :class:`Human` in a world per tracked person, placed at the person's
    frame.
    """

    world: World
    """
    The world the humans are kept in.
    """

    body_radius: float = 0.3
    """
    Radius of the cylinder standing in for a person's body.
    """

    body_height: float = 1.8
    """
    Height of that cylinder, which hangs down from the person's frame at head height.
    """

    humans: dict[str, Human] = field(default_factory=dict, init=False)
    """
    The mirrored humans by their ROS4HRI person ID.
    """

    def place(
        self, person_id: str, world_T_person: HomogeneousTransformationMatrix
    ) -> Human:
        """
        Put the person's human at ``world_T_person``, adding it first if it is new.

        :return: The person's human.
        """
        if person_id not in self.humans:
            self.humans[person_id] = self._add_human(person_id)
        self.humans[person_id].root.parent_connection.origin = world_T_person
        return self.humans[person_id]

    def keep_only(self, person_ids: list[str]) -> None:
        """
        Remove the humans of all persons not in ``person_ids``.
        """
        for person_id in [
            mirrored_id for mirrored_id in self.humans if mirrored_id not in person_ids
        ]:
            human = self.humans.pop(person_id)
            with self.world.modify_world():
                self.world.remove_semantic_annotation(human)
                self.world.remove_kinematic_structure_entity(human.root)

    def _add_human(self, person_id: str) -> Human:
        """
        Add a human for ``person_id`` at the world root, as a cylinder of a person's
        size on a free connection.

        :return: The new human.
        """
        body = Body(name=PrefixedName(f"person_{person_id}", prefix="ros4hri"))
        cylinder = Cylinder(width=2 * self.body_radius, height=self.body_height)
        cylinder.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=-self.body_height / 2, reference_frame=body
        )
        body.collision = ShapeCollection([cylinder], reference_frame=body)
        body.visual = body.collision
        human = Human(root=body)
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(body)
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=self.world.root, child=body, world=self.world
                )
            )
            self.world.add_semantic_annotation(human)
        return human

    def persons_in(self, area: ElevatorArea) -> list[str]:
        """
        :return: The IDs of the mirrored persons whose frame lies inside ``area``.
        """
        return [
            person_id
            for person_id, human in self.humans.items()
            if area.contains(Point3(reference_frame=human.root))
        ]


# %% elevator drive


@dataclass
class ElevatorDrive(LatestMessageSubscriber[JointState]):
    """
    The position of a simulated elevator's drive, read from its joint states.
    """

    joint_name: str = field(default="elevator_vertical_joint", kw_only=True)
    """
    Name of the drive joint in the joint states.
    """

    position_offset: float = field(default=0.0, kw_only=True)
    """
    Semantic drive position minus joint position.
    """

    position_tolerance: float = field(default=0.001, kw_only=True)
    """
    How far the received position may differ from the cabin's before the cabin is moved,
    so that an elevator standing still causes no world changes.
    """

    def apply_to(self, elevator: Elevator) -> None:
        """
        Move the elevator's cabin to the received drive position, if one was received
        and it differs from the cabin's by more than :attr:`position_tolerance`.
        """
        if (
            self.latest_message is None
            or self.joint_name not in self.latest_message.name
        ):
            return
        joint_position = self.latest_message.position[
            self.latest_message.name.index(self.joint_name)
        ]
        drive_position = joint_position + self.position_offset
        if (
            abs(drive_position - elevator.movable_joint.position)
            <= self.position_tolerance
        ):
            return
        elevator.movable_joint.position = drive_position


# %% publishing


@dataclass
class ElevatorPeople(HasROS2Node):
    """
    Mirrors the tracked persons into a world and publishes who is inside an elevator and
    who waits in front of it on the robot's level.
    """

    elevator: Elevator
    """
    The elevator whose areas are watched.
    """

    tracked_persons: TrackedPersons = field(kw_only=True)
    """
    The persons to watch.
    """

    mirror: PersonMirror = field(kw_only=True)
    """
    The mirrored persons.
    """

    tf: TFWrapper = field(kw_only=True)
    """
    Where the persons' and the robot's frames are looked up.
    """

    robot_frame: str = field(default=RobotFrame.BASE, kw_only=True)
    """
    TF frame of the robot's base, whose height decides the robot's level.
    """

    elevator_drive: ElevatorDrive | None = field(default=None, kw_only=True)
    """
    Where the cabin's position comes from, if it moves outside the world.
    """

    floor_tolerance: float = field(default=0.2, kw_only=True)
    """
    How far a level's floor may lie above the robot's base frame for the robot to count
    as standing on it.
    """

    publishers: dict[ElevatorPeopleTopic, Publisher] = field(init=False)
    """
    The publisher of each topic this class publishes.
    """

    def __post_init__(self) -> None:
        message_types = {
            ElevatorPeopleTopic.CABIN_PERSONS: IdsList,
            ElevatorPeopleTopic.CABIN_FOOTPRINT: PolygonStamped,
            ElevatorPeopleTopic.WAITING_AREA_PERSONS: IdsList,
            ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT: PolygonStamped,
        }
        self.publishers = {
            topic: self.node.create_publisher(message_type, topic, 1)
            for topic, message_type in message_types.items()
        }

    @classmethod
    def for_world(
        cls,
        node: Node,
        world: World,
        robot_frame: str,
        elevator_drive: ElevatorDrive | None,
    ) -> Self:
        """
        Watch the only elevator of ``world``.
        """
        return cls(
            node=node,
            elevator=world.get_semantic_annotations_by_type(Elevator)[0],
            tracked_persons=TrackedPersons(
                node=node, topic_name=Ros4HriTopic.TRACKED_PERSONS
            ),
            mirror=PersonMirror(world),
            tf=TFWrapper(node=node),
            robot_frame=robot_frame,
            elevator_drive=elevator_drive,
        )

    def update(self) -> None:
        """
        Mirror the persons and publish both areas with the persons inside them.
        """
        if self.elevator_drive is not None:
            self.elevator_drive.apply_to(self.elevator)
        self._mirror_tracked_persons()
        self._publish(
            ElevatorCabinArea(self.elevator),
            ElevatorPeopleTopic.CABIN_PERSONS,
            ElevatorPeopleTopic.CABIN_FOOTPRINT,
        )
        robot_level = self.robot_level()
        if robot_level is None:
            return
        self._publish(
            ElevatorWaitingArea(self.elevator, robot_level),
            ElevatorPeopleTopic.WAITING_AREA_PERSONS,
            ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT,
        )

    def _mirror_tracked_persons(self) -> None:
        """
        Make the mirror hold exactly the tracked persons whose frames TF knows, at those
        frames.
        """
        person_ids = self.tracked_persons.person_ids
        self.mirror.keep_only(person_ids)
        for person_id in person_ids:
            world_T_person = self._lookup(f"person_{person_id}")
            if world_T_person is not None:
                self.mirror.place(person_id, world_T_person)

    def robot_level(self) -> Level | None:
        """
        :return: The highest level whose floor is at most :attr:`floor_tolerance` above
            the robot's base, or None if the robot's frame is unknown or below every level.
        """
        world_T_robot = self._lookup(self.robot_frame)
        if world_T_robot is None:
            return None
        robot_height = float(world_T_robot.z)
        levels_below = [
            level
            for level in self.elevator._world.get_semantic_annotations_by_type(Level)
            if float(level.floor_plane[0].z) <= robot_height + self.floor_tolerance
        ]
        if not levels_below:
            return None
        return max(levels_below, key=lambda level: float(level.floor_plane[0].z))

    def _lookup(self, frame: str) -> HomogeneousTransformationMatrix | None:
        """
        :return: The pose of ``frame`` in the world root frame, or None if TF does not
            know it yet.
        """
        world = self.elevator._world
        if not self.tf.tf_buffer.can_transform(world.root.name.name, frame, Time()):
            return None
        transform = self.tf.lookup_transform(world.root.name.name, frame).transform
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=transform.translation.x,
            pos_y=transform.translation.y,
            pos_z=transform.translation.z,
            quat_x=transform.rotation.x,
            quat_y=transform.rotation.y,
            quat_z=transform.rotation.z,
            quat_w=transform.rotation.w,
            reference_frame=world.root,
        )

    def _publish(
        self,
        area: ElevatorArea,
        persons_topic: ElevatorPeopleTopic,
        footprint_topic: ElevatorPeopleTopic,
    ) -> None:
        """
        Publish the persons inside ``area`` and the area's footprint.
        """
        stamp = self.node.get_clock().now().to_msg()
        frame_id = self.elevator._world.root.name.name
        persons = IdsList(ids=self.mirror.persons_in(area))
        persons.header.stamp, persons.header.frame_id = stamp, frame_id
        self.publishers[persons_topic].publish(persons)
        footprint = PolygonStamped()
        footprint.header.stamp, footprint.header.frame_id = stamp, frame_id
        footprint.polygon.points = [
            Point32(x=float(corner.x), y=float(corner.y), z=float(corner.z))
            for corner in area.footprint()
        ]
        self.publishers[footprint_topic].publish(footprint)


# %% running against the Gazebo simulator


def main(rate: float = 5.0, robot_frame: str = RobotFrame.BASE) -> None:
    """
    Watch the elevator of the lab building in the Gazebo simulator.

    :param rate: Updates per second.
    :param robot_frame: TF frame of the simulated robot's base.
    """
    rclpy.init()
    node = rclpy.create_node("ra_fetch_elevator_people")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    world = LabBuilding().create_world()
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    elevator_drive = ElevatorDrive(
        node=node,
        topic_name=SimulatorTopic.ELEVATOR_JOINT_STATES,
        position_offset=elevator.drive_position_for_floor(
            world.get_semantic_annotations_by_type(GroundFloor)[0]
        ),
    )
    elevator_people = ElevatorPeople.for_world(node, world, robot_frame, elevator_drive)
    node.create_timer(1.0 / rate, elevator_people.update)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
