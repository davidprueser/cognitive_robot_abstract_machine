from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from threading import Thread

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig
from robokudo.exceptions import UnknownWorldFrameError
from robokudo.io.camera_interface import KinectCameraInterface
from robokudo.io.tf_listener_proxy import TFListenerProxy
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

from ._perception_world import FrameName, camera_pose, world_with_a_head_camera

# %% camera receiving frames


@pytest.fixture
def kinect_camera_interface(node: Node) -> Iterator[KinectCameraInterface]:
    """
    A camera interface receiving raw images, which looks its camera pose up in TF
    against the test frames, where TF does not hold the camera yet.
    """
    yield KinectCameraInterface(
        KinectCameraConfig(
            color_hints="raw",
            depth_hints="raw",
            hi_res_mode=False,
            tf_from=FrameName.CAMERA,
            tf_to=FrameName.WORLD,
        )
    )
    TFListenerProxy().clear_node(node)


@pytest.fixture
def camera_interface(
    kinect_camera_interface: KinectCameraInterface,
) -> KinectCameraInterface:
    """
    The camera interface of :func:`kinect_camera_interface`, where TF holds the camera
    at the world frame.
    """
    kinect_camera_interface.tf_buffer.set_transform_static(
        camera_transform_at(x=0.0, stamp=TimeMessage()), "test"
    )
    return kinect_camera_interface


def camera_transform_at(x: float, stamp: TimeMessage) -> TransformStamped:
    """
    :param x: Position of the camera along the world x axis.
    :param stamp: When the camera was there.
    :return: The transform of the camera in the world frame, looking along the world
        frame axes, as TF holds it.
    """
    world_T_camera = TransformStamped()
    world_T_camera.header.frame_id = FrameName.WORLD
    world_T_camera.header.stamp = stamp
    world_T_camera.child_frame_id = FrameName.CAMERA
    world_T_camera.transform.translation.x = x
    world_T_camera.transform.rotation.w = 1.0
    return world_T_camera


def receive_a_frame(
    camera_interface: KinectCameraInterface, received_at: timedelta = timedelta()
) -> None:
    """
    Hand ``camera_interface`` a color image, a depth image and their camera info, as its
    subscriptions do.

    :param camera_interface: The camera interface receiving the frame.
    :param received_at: When the color image was taken; zero means unknown.
    """
    bridge = CVBridgeWorkaround()
    color_image = bridge.cv2_to_imgmsg(np.zeros((4, 4, 3), dtype=np.uint8), "bgr8")
    color_image.header.stamp = Time(
        nanoseconds=int(received_at.total_seconds() * 1e9)
    ).to_msg()
    camera_interface.callback(
        color_image,
        bridge.cv2_to_imgmsg(np.zeros((4, 4), dtype=np.float32), "32FC1"),
        CameraInfo(width=4, height=4, k=[1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]),
    )


def world_without_the_world_frame() -> World:
    """
    :return: A world whose only body is not the world frame the camera is measured
        against.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root_of_another_world")))
    return world


# %% failed camera placement


class TestFailedCameraPlacement:
    def test_camera_keeps_receiving_after_its_pose_could_not_be_placed(
        self, camera_interface: KinectCameraInterface
    ) -> None:
        receive_a_frame(camera_interface)
        with pytest.raises(UnknownWorldFrameError):
            camera_interface.set_data(CAS(), world_without_the_world_frame())

        receiving = Thread(
            target=receive_a_frame, args=(camera_interface,), daemon=True
        )
        receiving.start()
        receiving.join(timeout=timedelta(seconds=5).total_seconds())

        assert not receiving.is_alive()


# %% camera pose


class TestCameraPose:
    def test_camera_the_world_holds_is_placed_where_the_world_holds_it(
        self, kinect_camera_interface: KinectCameraInterface
    ) -> None:
        world = world_with_a_head_camera()
        cas = CAS()
        receive_a_frame(kinect_camera_interface)

        kinect_camera_interface.set_data(cas, world)

        assert np.allclose(
            cas.camera_to_world_transform.to_np(),
            world.get_body_by_name(FrameName.CAMERA).global_transform.to_np(),
        )

    def test_other_camera_is_placed_where_tf_held_it_when_the_image_was_taken(
        self, kinect_camera_interface: KinectCameraInterface
    ) -> None:
        image_taken_at = timedelta(seconds=1)
        for stamp, x in ((image_taken_at, 1.0), (timedelta(seconds=2), 2.0)):
            kinect_camera_interface.tf_buffer.set_transform(
                camera_transform_at(
                    x=x,
                    stamp=Time(nanoseconds=int(stamp.total_seconds() * 1e9)).to_msg(),
                ),
                "test",
            )
        cas = CAS()
        receive_a_frame(kinect_camera_interface, received_at=image_taken_at)

        kinect_camera_interface.set_data(cas, World())

        assert np.allclose(
            cas.camera_to_world_transform.to_np(), camera_pose(x=1.0).to_np()
        )
