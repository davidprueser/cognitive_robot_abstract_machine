"""
ROS camera interface for RGB-only cameras in RoboKudo.

This module provides an interface for ROS cameras that only provide RGB images
without depth information. It supports:

* Synchronized RGB image and camera info subscription
* Image rotation with intrinsics adjustment
* Placing the camera in the world perception perceives into
* Thread-safe data handling

The module is used for cameras like:

* Hand cameras on robotic arms
* Standard USB cameras with ROS drivers
* Network cameras with ROS interfaces
"""

from __future__ import annotations

import copy
from threading import Lock

import builtin_interfaces.msg
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from typing_extensions import Any, TYPE_CHECKING

from robokudo.cas import CAS, CASViews
from robokudo.io.camera_interface import ROSCameraInterface
from robokudo.types.cv import ColorToDepthRatio
from robokudo.utils.type_conversion import o3d_camera_intrinsics_from_ros_camera_info

if TYPE_CHECKING:
    import numpy.typing as npt
    from semantic_digital_twin.world import World

# %% ROS camera without depth


class ROSCameraWithoutDepthInterface(ROSCameraInterface):
    """
    A ROS camera interface for RGB-only cameras.

    This class handles cameras that provide only RGB images without depth data. It
    synchronizes RGB images with camera calibration information and rotates them, with
    their intrinsics, as configured.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the RGB-only camera interface.

        Sets up ROS subscribers and synchronization for RGB image and camera info
        topics.

        :param camera_config: Configuration for the camera interface
        """
        super().__init__(camera_config)

        self.color_subscriber: Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_color_configured() else Image,
            camera_config.topic_color,
        )
        """
        Subscriber for RGB image topic.
        """

        self.camera_info_subscriber: Subscriber = Subscriber(
            self.node, CameraInfo, camera_config.topic_camera_info
        )
        """
        Subscriber for camera info topic.
        """
        ts = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.camera_info_subscriber],
            queue_size=10,
            slop=0.4,
        )
        ts.registerCallback(self.callback)

        self.rk_logger.info("Subscribed to: ")
        self.rk_logger.info(f"  {camera_config.topic_color}")
        self.rk_logger.info(f"  {camera_config.topic_camera_info}")

        self.color: npt.NDArray | None = None
        """
        Latest RGB image data.
        """

        self.camera_info: CameraInfo | None = None
        """
        Latest camera calibration info.
        """

        self.timestamp: builtin_interfaces.msg.Time | None = None
        """
        Timestamp of latest data.
        """

        self.lock: Lock = Lock()
        """
        Thread synchronization lock.
        """

    def callback(
        self, color_data: Image | CompressedImage, camera_info: CameraInfo
    ) -> None:
        """
        Keep a color image and the camera info received with it as the latest data.

        :param color_data: RGB image message
        :param camera_info: Camera calibration info message
        """
        with self.lock:
            self.color = self.decode_color_image(color_data)
            self.timestamp = color_data.header.stamp
            self.camera_info = camera_info
            self._has_new_data = True

    def set_data(self, cas: CAS, world: World) -> None:
        """
        :meth:`CameraInterface.set_data` for the latest color image, rotated as
        configured. The camera info and intrinsics in ``cas`` describe the rotated
        image.
        """
        if not self.has_new_data():
            return

        with self.lock:
            color = self.color
            camera_info = self.camera_info
            camera_intrinsic = o3d_camera_intrinsics_from_ros_camera_info(camera_info)
            image_rotation = self.camera_config.image_rotation
            if image_rotation is not None:
                color = image_rotation.rotate(color)
                camera_intrinsic = image_rotation.rotate_intrinsics(camera_intrinsic)
                camera_info = copy.deepcopy(camera_info)
                camera_info.width = camera_intrinsic.width
                camera_info.height = camera_intrinsic.height
                camera_info.k = camera_intrinsic.intrinsic_matrix.flatten().tolist()

            cas.set(CASViews.COLOR_IMAGE, color)
            cas.set(CASViews.DEPTH_IMAGE, None)
            cas.set(CASViews.CAMERA_INFO, camera_info)
            cas.set(CASViews.CAMERA_INTRINSIC, camera_intrinsic)
            cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
            self.place_camera(cas, world, self.timestamp)

            self._has_new_data = False
