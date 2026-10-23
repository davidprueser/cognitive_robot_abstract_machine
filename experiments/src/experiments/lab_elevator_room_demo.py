"""
Spawn the lab elevator room traced from the lab occupancy grid map and show it in RViz.

Run with ROS 2 sourced, then in RViz set the fixed frame to ``map`` and add a
MarkerArray display on ``/semworld/viz_marker`` with durability *Transient Local*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.predefined_maps.lab_elevator_room import LabElevatorRoom
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

MAP_FRAME = "map"
"""
Name of the world root, matching the frame the lab occupancy grid map is published in.
"""

NODE_NAME = "lab_elevator_room_demo"
"""
Name of the ROS 2 node the demo publishes on.
"""


def create_lab_elevator_room_world() -> World:
    """
    :return: A world rooted in the map frame holding the lab elevator room, with every
        door open.
    """
    world = World.create_with_root_body(MAP_FRAME)
    LabElevatorRoom().spawn(world, "lab")
    for door in world.get_semantic_annotations_by_type(Door):
        door.movable_joint.position = door.movable_joint.dof.limits.upper.position
    return world


@dataclass
class RVizWorldPublisher:
    """
    Publishes a world to RViz: its tf tree and its bodies as latched markers.

    Both are re-sent periodically: the tf tree so it does not expire from RViz's tf
    buffer while the world stands still, and the markers so that an RViz started after
    the publisher resolves their frames once the tf tree has reached it.
    """

    world: World
    """
    The world to show.
    """

    node: Node
    """
    The node the tf tree and markers are published on.
    """

    tf_period_seconds: float = 0.5
    """
    How often the tf tree is re-sent.
    """

    marker_period_seconds: float = 2.0
    """
    How often the markers are re-sent.
    """

    tf_publisher: TFPublisher = field(init=False)
    """
    Publishes the tf tree of :attr:`world`.
    """

    marker_publisher: VizMarkerPublisher = field(init=False)
    """
    Publishes the bodies of :attr:`world` as markers.
    """

    def __post_init__(self):
        self.tf_publisher = TFPublisher(_world=self.world, node=self.node)
        self.marker_publisher = VizMarkerPublisher(
            _world=self.world, node=self.node, tf_publisher=self.tf_publisher
        )
        self.node.create_timer(
            self.tf_period_seconds, self.tf_publisher.on_state_change
        )
        self.node.create_timer(
            self.marker_period_seconds, self.marker_publisher.on_model_change
        )


def main() -> None:
    """
    Publish the lab elevator room to RViz until interrupted.
    """
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = rclpy.create_node(NODE_NAME)
    world = create_lab_elevator_room_world()
    RVizWorldPublisher(world=world, node=node)
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
