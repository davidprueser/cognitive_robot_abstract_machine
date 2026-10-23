"""
Tests for the camera interface that receives the color, depth and camera info topics of
a Kinect.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo

from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig
from robokudo.io.camera_interface import KinectCameraInterface
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

# %% camera pose


def camera_pose_at(
    interface: KinectCameraInterface, stamp: Time, x: float
) -> TransformStamped:
    """
    :param interface: The interface whose camera is posed.
    :param stamp: When the camera had the pose.
    :param x: How far the camera is from the world frame of ``interface``, along its x
        axis.
    :return: The pose of the camera of ``interface`` at ``stamp``.
    """
    world_T_camera = TransformStamped()
    world_T_camera.header.stamp = stamp.to_msg()
    world_T_camera.header.frame_id = interface.tf_to
    world_T_camera.child_frame_id = interface.tf_from
    world_T_camera.transform.translation.x = x
    world_T_camera.transform.rotation.w = 1.0
    return world_T_camera


def test_camera_pose_is_the_one_at_the_stamp_of_the_color_image(node: Node) -> None:
    interface = KinectCameraInterface(
        KinectCameraConfig(color_hints="raw", depth_hints="raw")
    )
    recording_time = Time(seconds=1)
    interface.tf_buffer.set_transform(
        camera_pose_at(interface, recording_time, x=1.0), "test"
    )
    interface.tf_buffer.set_transform(
        camera_pose_at(interface, Time(seconds=2), x=2.0), "test"
    )
    bridge = CVBridgeWorkaround()
    color = bridge.cv2_to_imgmsg(np.zeros((960, 1280, 3), np.uint8), "bgr8")
    color.header.stamp = recording_time.to_msg()
    depth = bridge.cv2_to_imgmsg(np.zeros((480, 640), np.float32), "32FC1")

    interface.callback(color, depth, CameraInfo())

    assert interface.camera_translation == [1.0, 0.0, 0.0]
