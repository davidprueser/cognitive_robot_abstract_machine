from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from rclpy.node import Node
from semantic_digital_twin.world import World
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_hsr_handcam import (
    HsrHandCameraConfig,
)
from robokudo.io.ros_camera_without_depth_interface import (
    ROSCameraWithoutDepthInterface,
)
from robokudo.io.tf_listener_proxy import TFListenerProxy
from robokudo.types.cv import ImageRotation
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

from ._perception_world import FrameName, world_with_a_head_camera

# %% camera receiving frames


def hand_camera_interface(
    image_rotation: ImageRotation | None = None, lookup_viewpoint: bool = False
) -> ROSCameraWithoutDepthInterface:
    """
    :param image_rotation: How the camera interface rotates its images.
    :param lookup_viewpoint: Whether the camera interface places its camera.
    :return: A camera interface receiving raw color images of a camera without depth,
        placed against the test frames.
    """
    return ROSCameraWithoutDepthInterface(
        HsrHandCameraConfig(
            color_hints="raw",
            image_rotation=image_rotation,
            lookup_viewpoint=lookup_viewpoint,
            tf_from=FrameName.CAMERA,
            tf_to=FrameName.WORLD,
        )
    )


@pytest.fixture
def placing_hand_camera_interface(
    node: Node,
) -> Iterator[ROSCameraWithoutDepthInterface]:
    """
    A camera interface of :func:`hand_camera_interface` that places its camera.
    """
    yield hand_camera_interface(lookup_viewpoint=True)
    TFListenerProxy().clear_node(node)


def color_image() -> npt.NDArray:
    """
    :return: A color image 8 pixels wide and 6 pixels high, whose pixels all differ.
    """
    return np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))


def camera_info() -> CameraInfo:
    """
    :return: The camera info of :func:`color_image`, whose focal lengths and principal
        point coordinates all differ.
    """
    return CameraInfo(
        width=8, height=6, k=[1.0, 0.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0]
    )


def cas_of_a_received_frame(
    camera_interface: ROSCameraWithoutDepthInterface, world: World
) -> CAS:
    """
    Hand ``camera_interface`` :func:`color_image` and :func:`camera_info`, as its
    subscriptions do, and put them into a CAS.

    :param camera_interface: The camera interface receiving the frame.
    :param world: The world perception perceives into.
    :return: The CAS holding the frame.
    """
    camera_interface.callback(
        CVBridgeWorkaround().cv2_to_imgmsg(color_image(), "bgr8"), camera_info()
    )
    cas = CAS()
    camera_interface.set_data(cas, world)
    return cas


# %% frame without depth


class TestFrameWithoutDepth:
    def test_color_image_is_put_into_the_cas(self, node: Node) -> None:
        cas = cas_of_a_received_frame(hand_camera_interface(), World())

        assert np.array_equal(cas.color_image, color_image())

    def test_cas_holds_no_depth_image(self, node: Node) -> None:
        cas = cas_of_a_received_frame(hand_camera_interface(), World())

        assert cas.depth_image is None


# %% image rotation


class TestImageRotation:
    def test_image_is_rotated(self, node: Node) -> None:
        cas = cas_of_a_received_frame(
            hand_camera_interface(
                image_rotation=ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE
            ),
            World(),
        )

        assert np.array_equal(cas.color_image, np.rot90(color_image()))

    def test_intrinsics_are_rotated_with_the_image(self, node: Node) -> None:
        cas = cas_of_a_received_frame(
            hand_camera_interface(
                image_rotation=ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE
            ),
            World(),
        )

        assert np.array_equal(
            cas.camera_intrinsic.intrinsic_matrix,
            ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE.rotate_intrinsics(
                cas_of_a_received_frame(
                    hand_camera_interface(), World()
                ).camera_intrinsic
            ).intrinsic_matrix,
        )

    def test_camera_info_matches_the_rotated_intrinsics(self, node: Node) -> None:
        cas = cas_of_a_received_frame(
            hand_camera_interface(
                image_rotation=ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE
            ),
            World(),
        )

        assert (cas.camera_info.width, cas.camera_info.height) == (
            cas.camera_intrinsic.width,
            cas.camera_intrinsic.height,
        )
        assert np.array_equal(
            np.reshape(cas.camera_info.k, (3, 3)), cas.camera_intrinsic.intrinsic_matrix
        )


# %% camera pose


class TestCameraPose:
    def test_camera_the_world_holds_is_placed_where_the_world_holds_it(
        self, placing_hand_camera_interface: ROSCameraWithoutDepthInterface
    ) -> None:
        world = world_with_a_head_camera()

        cas = cas_of_a_received_frame(placing_hand_camera_interface, world)

        assert np.allclose(
            cas.camera_to_world_transform.to_np(),
            world.get_body_by_name(FrameName.CAMERA).global_transform.to_np(),
        )
