"""
Who is inside the elevator and who waits in front of it, from the people a ROS4HRI
(REP-155) pipeline tracks.

Tracked persons are mirrored into the semantic digital twin as :class:`Human` bodies at
their ``person_<personID>`` TF frame. Who of them is inside the cabin and who is in the
waiting area in front of its doors on the level the robot is on is published as
``hri_msgs/IdsList`` of person IDs, along with both areas' outlines.

Running :func:`main` next to a controller of the TIAGo ARISE Gazebo simulator with its
``tiago_arise_humans`` launch gives the real data flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import rclpy
from geometry_msgs.msg import Point32, PolygonStamped
from hri_msgs.msg import IdsList
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from rclpy.time import Time

from experiments.tiago_arise_gazebo.elevator_drive import ElevatorDrive, SimulatorTopic
from semantic_digital_twin.adapters.ros.input_synchronization import (
    TfFrameSynchronizer,
    TopicInputSynchronizer,
)
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeBuilding
from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Human,
    Level,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region

# %% topics


class Ros4HriTopic(StrEnum):
    """
    Topics of REP-155 this module reads.
    """

    TRACKED_PERSONS = "/humans/persons/tracked"


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
class TrackedPersons(TopicInputSynchronizer[IdsList]):
    """
    Keeps one :class:`Human` in the world per person a ROS4HRI person manager tracks, at
    the person's TF frame, for as long as TF knows that frame.
    """

    humans: dict[str, Human] = field(default_factory=dict, init=False)
    """
    The humans of the tracked persons by their ROS4HRI person ID.
    """

    person_frames: TfFrameSynchronizer = field(init=False)
    """
    Moves each human to its person's frame.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.person_frames = TfFrameSynchronizer(world=self.world, node=self.node)

    def apply_message(self, message: IdsList) -> None:
        """
        Make the world hold exactly the humans of the tracked persons whose frames TF
        knows, at those frames.
        """
        located_ids = [
            person_id
            for person_id in message.ids
            if self.person_frames.tf_wrapper.wait_for_transform(
                self.world.root.name.name,
                self.frame_of(person_id),
                Time(),
                Duration(),
            )
        ]
        for person_id in [
            person_id for person_id in self.humans if person_id not in located_ids
        ]:
            self._remove_human(person_id)
        for person_id in located_ids:
            if person_id not in self.humans:
                self.humans[person_id] = self._add_human(person_id)
        self.person_frames.apply()

    def close(self) -> None:
        """
        Unlike :meth:`InputSynchronizer.close`, also removes the humans of all persons
        from the world.
        """
        for person_id in list(self.humans):
            self._remove_human(person_id)
        super().close()

    @staticmethod
    def frame_of(person_id: str) -> str:
        """
        :return: The TF frame REP-155 publishes the person with ``person_id`` at.
        """
        return f"person_{person_id}"

    def _add_human(
        self, person_id: str, body_radius: float = 0.3, body_height: float = 1.8
    ) -> Human:
        """
        Add a human for ``person_id`` at the world root, as a cylinder of a person's
        size on a connection that follows the person's frame.

        :param body_radius: Radius of the cylinder standing in for the person's body.
        :param body_height: Height of that cylinder, which hangs down from the person's
            frame at head height.
        :return: The new human.
        """
        body = Body(name=PrefixedName(self.frame_of(person_id), prefix="ros4hri"))
        cylinder = Cylinder(width=2 * body_radius, height=body_height)
        cylinder.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=-body_height / 2, reference_frame=body
        )
        body.collision = ShapeCollection([cylinder], reference_frame=body)
        body.visual = body.collision
        human = Human(name=PrefixedName(person_id, prefix="ros4hri"), root=body)
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(body)
            connection = Connection6DoF.create_with_dofs(
                parent=self.world.root, child=body, world=self.world
            )
            self.world.add_connection(connection)
            self.world.add_semantic_annotation(human)
        self.person_frames.track(
            connection, self.world.root.name.name, self.frame_of(person_id)
        )
        return human

    def _remove_human(self, person_id: str) -> None:
        """
        Remove the human of ``person_id`` from the world.
        """
        human = self.humans.pop(person_id)
        self.person_frames.untrack(human.root.parent_connection)
        with self.world.modify_world():
            self.world.remove_semantic_annotation(human)
            self.world.remove_kinematic_structure_entity(human.root)


# %% publishing


@dataclass
class ElevatorPeoplePublisher(HasROS2Node):
    """
    Publishes which of the tracked persons are inside an elevator's cabin and which wait
    in front of it on the robot's level, with the outlines of both areas.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator whose areas are published.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot whose base's height decides on which level people count as waiting.
    """

    tracked_persons: TrackedPersons = field(kw_only=True)
    """
    The persons to publish.
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

    def publish(self) -> None:
        """
        Publish the cabin with the humans inside it and, unless the robot stands below
        every level, the waiting area on the robot's level with the humans inside it.
        """
        self._publish_area(
            self.elevator.cabin_area,
            ElevatorPeopleTopic.CABIN_PERSONS,
            ElevatorPeopleTopic.CABIN_FOOTPRINT,
        )
        robot_level = Level.under(self.robot.root, self.floor_tolerance)
        if robot_level is None:
            return
        self._publish_area(
            self.elevator.waiting_areas[robot_level],
            ElevatorPeopleTopic.WAITING_AREA_PERSONS,
            ElevatorPeopleTopic.WAITING_AREA_FOOTPRINT,
        )

    def _publish_area(
        self,
        area: Region,
        persons_topic: ElevatorPeopleTopic,
        footprint_topic: ElevatorPeopleTopic,
    ) -> None:
        """
        Publish the IDs of the tracked persons whose humans stand inside ``area`` and
        the area's footprint.
        """
        stamp = self.node.get_clock().now().to_msg()
        frame_id = area._world.root.name.name
        persons = IdsList(
            ids=[
                person_id
                for person_id, human in self.tracked_persons.humans.items()
                if InsideRegion(human.root, area)()
            ]
        )
        persons.header.stamp, persons.header.frame_id = stamp, frame_id
        self.publishers[persons_topic].publish(persons)
        footprint = PolygonStamped()
        footprint.header.stamp, footprint.header.frame_id = stamp, frame_id
        footprint.polygon.points = [
            Point32(x=float(corner.x), y=float(corner.y), z=float(corner.z))
            for corner in self._footprint_of(area)
        ]
        self.publishers[footprint_topic].publish(footprint)

    @staticmethod
    def _footprint_of(area: Region) -> list[Point3]:
        """
        :return: The corners of ``area``'s outline on its floor, counter-clockwise, in
            the world root frame.
        """
        min_point, max_point = area.area.min_point, area.area.max_point
        return [
            area._world.transform(
                Point3(x, y, min_point.z, reference_frame=area), area._world.root
            )
            for x, y in (
                (min_point.x, min_point.y),
                (max_point.x, min_point.y),
                (max_point.x, max_point.y),
                (min_point.x, max_point.y),
            )
        ]


# %% running against the Gazebo simulator


def main(rate: float = 5.0) -> None:
    """
    Publish who is where around the elevator of the ARISE Gazebo office in the world a
    running controller serves, spawning the office into it if it is not there yet.

    :param rate: Updates per second.
    """
    rclpy.init()
    node = rclpy.create_node("ra_fetch_elevator_people")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    world = fetch_world_from_service(node)
    WorldSynchronizer(_world=world, node=node)
    office = AriseOfficeBuilding()
    if not world.get_semantic_annotations_by_type(Elevator):
        office.spawn(world)
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    world_inputs = WorldStateInputs(world=world)
    world_inputs.add_input(
        ElevatorDrive(
            world=world,
            node=node,
            topic_name=SimulatorTopic.ELEVATOR_JOINT_STATES,
            elevator=elevator,
            joint_name=office.simulated_drive_joint_name(elevator),
        )
    )
    tracked_persons = TrackedPersons(
        world=world, node=node, topic_name=Ros4HriTopic.TRACKED_PERSONS
    )
    world_inputs.add_input(tracked_persons)
    elevator_people_publisher = ElevatorPeoplePublisher(
        node=node,
        elevator=elevator,
        robot=world.get_semantic_annotations_by_type(AbstractRobot)[0],
        tracked_persons=tracked_persons,
    )
    node.create_timer(1.0 / rate, world_inputs.synchronize)
    node.create_timer(1.0 / rate, elevator_people_publisher.publish)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
