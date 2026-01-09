import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from std_msgs.msg import ColorRGBA
from typing_extensions import List, Any, Dict
from visualization_msgs.msg import MarkerArray, Marker

from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from geometry_msgs.msg import (
    TransformStamped,
    Vector3,
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)


@dataclass
class PosePublisher(StateChangeCallback):
    pose: HomogeneousTransformationMatrix
    """
    The pose to publish.
    """
    node: rclpy.Node
    """
    ROS node handle, used to create the publisher.
    """
    topic_name: str = "/semworld/viz_marker"
    """
    Topic name to publish the pose marker on.
    """
    publish_chain: bool = False
    """
    If true publishes all transforms between map and pose
    """

    publisher: Any = field(init=False)

    def _notify(self):
        world = self.pose.reference_frame._world
        global_pose = world.transform(self.pose, self.world.root)
        marker_array = self._create_marker_array(global_pose)
        self.publisher.publish(marker_array)

    def __post_init__(self):
        self.publisher = self.node.create_publisher(MarkerArray, self.topic_name, 10)
        time.sleep(0.2)
        self._notify()

    def _create_marker_array(
        self, global_pose: HomogeneousTransformationMatrix
    ) -> MarkerArray:
        """
        Creates a MarkerArray to visualize a Pose in RViz. The pose is visualized as an arrow for each axis to represent
        the position and orientation of the pose.
        :param global_pose: The pose to in global frame
        """
        marker_array = MarkerArray()
        for i in range(3):
            axis = [0, 0, 0]
            axis[i] = 1
            color = [0, 0, 0, 1]
            color[i] = 1
            position = global_pose.to_position().to_np()[:3]
            orientation = global_pose.to_rotation_matrix().to_quaternion().to_np()

            end_point = Point(
                **dict(zip(["x", "y", "z"], np.array(position) + np.array(axis)))
            )
            p = Pose(
                position=Vector3(**dict(zip(["x", "y", "z"], axis))),
                orientation=Quaternion(**dict(zip(["x", "y", "z", "w"], orientation))),
            )
            c = ColorRGBA(**dict(zip(["r", "g", "b", "a"], color)))

            marker_array.markers.append(
                self._create_marker(
                    c, i, p, Point(**dict(zip(["x", "y", "z"], position))), end_point
                )
            )
        return marker_array

    def _create_marker(
        self,
        color: ColorRGBA,
        _id: int,
        pose: Pose,
        start_point: Point,
        end_point: Point,
    ) -> Marker:
        """
        Creates a visualization marker for one axis of the pose.
        :param color: The color of the axis.
        :param _id: The id of the axis to identify the arrow.
        :param pose: The pose to publish
        :param start_point: The start point of the arrow.
        :param end_point: The end point of the arrow.
        """
        m = Marker()
        m.action = Marker.ADD
        m.type = Marker.ARROW
        m.id = _id
        m.header.frame_id = "map"
        m.pose = pose

        m.points = [start_point, end_point]

        m.scale = Vector3(x=0.1, y=0.1, z=0.1)
        m.color = color
        m.ns = self.pose.reference_frame.name.name

        return m
