"""
Camera interface module for RoboKudo.

This module provides base classes and implementations for interfacing with
various camera types in RoboKudo. It supports:

* ROS camera interfaces (raw and compressed)
* Kinect-style RGB-D cameras
* Camera calibration handling
* Transform lookups
* Synchronized data acquisition
* Thread-safe operation

The module handles:

* RGB and depth image acquisition
* Camera calibration information
* Camera-to-world transforms
* Data synchronization
* Format conversions
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from threading import Lock

import builtin_interfaces.msg
import cv2
import message_filters
import numpy as np
import open3d as o3d
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from tf2_ros import Buffer
from typing_extensions import Optional, Any, TYPE_CHECKING, Union

from robokudo.io.camera_placement import (
    CameraPlacement,
    CameraPoseMeasurement,
    MeasuredCameraPose,
)
from robokudo.cas import CASViews, CAS
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import CameraDataMissing
from robokudo.io.tf_listener_proxy import TFListenerProxy
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.types.cv import ColorToDepthRatio


# %% camera interface


class CameraInterface(object):
    """
    Base class for all camera interfaces in RoboKudo.

    This class defines the basic interface that all camera implementations must provide.
    It handles configuration and data availability tracking.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the camera interface.

        :param camera_config: Configuration for the camera
        """
        self._has_new_data: bool = False
        """
        Whether new data is available.
        """
        self.camera_config: Any = camera_config
        """
        Camera configuration object.
        """
        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """
        RoboKudo logger instance.
        """

    def has_new_data(self) -> bool:
        """
        Check if new data is available.

        :return: True if new data is available, False otherwise
        """
        return self._has_new_data

    def place_static_camera(self, cas: CAS, world: World) -> None:
        """
        Place the camera of ``cas`` in ``world`` at the pose the configuration fixes, if
        it fixes one.

        :param cas: The CAS of the data the camera recorded.
        :param world: The world perception perceives into.
        """
        if not self.camera_config.static_camera_transform_enabled:
            return
        CameraPlacement(world).place(
            cas,
            MeasuredCameraPose(
                world_frame=self.camera_config.static_world_frame,
                camera_frame=self.camera_config.static_camera_frame,
                world_T_camera=self.camera_config.static_world_T_camera,
            ),
        )

    def set_data(self, cas: CAS, world: World) -> None:
        """
        This method is supposed to read in, convert (if needed) and put the data into
        the CAS. If you are running a CameraInterface which is getting data via callback
        methods, please make sure to keep callbacks light and do the main conversion
        work here! Callbacks should be short.

        The camera is placed in ``world`` where it was when it recorded the data, if
        that is known.

        :param cas: The CAS where the data should be placed in
        :param world: The world perception perceives into
        """
        raise NotImplementedError


# %% camera pose in TF


@dataclass
class CameraPoseLookup(CameraPoseMeasurement):
    """
    Measures a camera's pose by looking up where TF held it at :attr:`time`.
    """

    tf_buffer: Buffer
    """
    The TF buffer the pose is looked up in.
    """

    time: Time
    """
    When the camera was at the pose.
    """

    timeout: Duration = field(default_factory=lambda: Duration(seconds=1.0 / 30.0))
    """
    How long to wait for TF to hold the pose.
    """

    def measure(self) -> HomogeneousTransformationMatrix:
        """
        :meth:`CameraPoseMeasurement.measure`, looked up in TF.

        :raises tf2_ros.TransformException: If TF does not hold the pose within
            :attr:`timeout`.
        """
        transform = self.tf_buffer.lookup_transform(
            target_frame=self.world_frame,
            source_frame=self.camera_frame,
            time=self.time,
            timeout=self.timeout,
        ).transform
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=transform.translation.x,
            pos_y=transform.translation.y,
            pos_z=transform.translation.z,
            quat_x=transform.rotation.x,
            quat_y=transform.rotation.y,
            quat_z=transform.rotation.z,
            quat_w=transform.rotation.w,
        )


# %% ROS camera


class ROSCameraInterface(CameraInterface):
    """
    Base class for ROS-based camera interfaces.

    This class extends the base camera interface with ROS-specific functionality like
    placing the camera at its pose in TF.
    """

    def __init__(self, camera_config: Any, node: Optional[Node] = None) -> None:
        """
        Initialize the ROS camera interface.

        :param camera_config: Configuration for the ROS camera
        :param node: A ROS node for transform lookups
        """
        super().__init__(camera_config)

        self.node = node if node is not None else ROSNodeRegistry().get()
        """
        ROS node for communication with ROS.
        """
        if hasattr(self.camera_config, "lookup_viewpoint"):
            self.lookup_viewpoint: bool = self.camera_config.lookup_viewpoint

            self.tf_from: str = camera_config.tf_from
            """
            Transform source frame.
            """
            self.tf_to: str = camera_config.tf_to
            """
            Transform target frame.
            """
        else:
            self.lookup_viewpoint: bool = False
            """
            Whether to look up camera transforms.
            """
        if self.lookup_viewpoint:
            self.tf_buffer: Buffer = TFListenerProxy().buffer_for_node(self.node)
            """
            TF buffer populated by the per-node transform listener.
            """

    def place_camera(
        self, cas: CAS, world: World, timestamp: builtin_interfaces.msg.Time
    ) -> None:
        """
        Place the camera of ``cas`` in ``world`` for the data received at ``timestamp``,
        which is stored in ``cas`` as the time of the data. A camera ``world`` holds is
        placed where ``world`` holds it, any other camera where TF held it at
        ``timestamp``. A camera that does not look up its pose is not placed.

        :param cas: The CAS of the data.
        :param world: The world perception perceives into.
        :param timestamp: When the data was received.
        :raises tf2_ros.TransformException: If the camera has to be looked up in TF and
            TF does not hold it at ``timestamp``.
        """
        if not self.lookup_viewpoint:
            return
        cas.data_timestamp = timestamp.sec * 1_000_000_000 + timestamp.nanosec
        CameraPlacement(world).place(
            cas,
            CameraPoseLookup(
                world_frame=self.tf_to,
                camera_frame=self.tf_from,
                tf_buffer=self.tf_buffer,
                time=Time.from_msg(timestamp),
            ),
        )


# %% Kinect camera


def depth_convert_workaround(msg: CompressedImage) -> npt.NDArray:
    """
    Convert compressed depth image to proper depth format.

    This is a workaround for handling compressed depth images in ROS.
    Source: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/

    :param msg: Compressed depth image message
    :return: Depth image as numpy array
    :raises Exception: If compression type is wrong or decoding fails
    """
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(";")
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if "compressedDepth" not in compr_type:
        raise Exception(
            "Compression type is not 'compressedDepth'."
            "You probably subscribed to the wrong topic."
        )

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(
        np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
    )
    # replaced np.fromstring with np.frombuffer because np.fromstring is deprecated in newer versions of numpy
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception(
            "Could not decode compressed depth image."
            "You may need to change 'depth_header_size'!"
        )

    if depth_fmt == "16UC1":
        # write raw image data
        return depth_img_raw
    elif depth_fmt == "32FC1":
        raw_header = msg.data[:depth_header_size]
        # header: int, float, float
        [compfmt, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
        depth_img_scaled = depthQuantA / (
            depth_img_raw.astype(np.float32) - depthQuantB
        )
        # filter max values
        depth_img_scaled[depth_img_raw == 0] = 0

        # depth_img_scaled provides distance in meters as f32
        # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        depth_img_mm = (depth_img_scaled * 1000).astype(np.uint16)
        return depth_img_mm
    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


class KinectCameraInterface(ROSCameraInterface):
    """
    Interface for Kinect-style RGB-D cameras using ROS.

    This class implements a camera interface for RGB-D cameras that publish color and
    depth images through ROS topics. It supports both raw and compressed image formats.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the Kinect camera interface.

        Sets up ROS subscribers and synchronization for color, depth, and camera info
        topics.

        :param camera_config: Configuration for the Kinect camera
        """
        super().__init__(camera_config)

        self.color_subscriber: message_filters.Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_color_configured() else Image,
            camera_config.topic_color,
        )
        """
        Color image subscriber.
        """
        self.depth_subscriber: message_filters.Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_depth_configured() else Image,
            camera_config.topic_depth,
        )
        """
        Depth image subscriber.
        """
        self.camera_info_subscriber: message_filters.Subscriber = Subscriber(
            self.node, CameraInfo, camera_config.topic_camera_info
        )
        """
        Camera info subscriber.
        """
        ts = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber, self.camera_info_subscriber],
            queue_size=10,
            slop=0.4,
        )
        ts.registerCallback(self.callback)

        self.rk_logger.info("Subscribed to: ")
        self.rk_logger.info(f"  {camera_config.topic_color}")
        self.rk_logger.info(f"  {camera_config.topic_depth}")
        self.rk_logger.info(f"  {camera_config.topic_camera_info}")

        self.color: Optional[npt.NDArray] = None
        """
        Latest color image.
        """
        self.depth: Optional[npt.NDArray] = None
        """
        Latest depth image.
        """
        self.camera_info: Optional[CameraInfo] = None
        """
        Latest camera info message.
        """
        self.camera_intrinsic: Optional[o3d.camera.PinholeCameraIntrinsic] = None
        """
        Open3D camera intrinsics.
        """
        self.color_to_depth_ratio: ColorToDepthRatio | None = None
        """
        Ratio between color and depth image sizes.
        """
        self.timestamp: Optional[builtin_interfaces.msg.Time] = None
        """
        Latest message timestamp.
        """
        self.lock: Lock = Lock()
        """
        Thread synchronization lock.
        """
        self.bridge: CVBridgeWorkaround = CVBridgeWorkaround()
        """
        NumPy-compatible replacement for cv_bridge.
        """

    def compressed_depth_configured(self) -> bool:
        """
        Check if compressed depth images are configured.

        :return: True if compressed depth is configured, False otherwise
        """
        return (
            hasattr(self.camera_config, "depth_hints")
            and self.camera_config.depth_hints == "compressedDepth"
        )

    def compressed_color_configured(self) -> bool:
        """
        Check if compressed color images are configured.

        :return: True if compressed color is configured, False otherwise
        """
        return (
            hasattr(self.camera_config, "color_hints")
            and self.camera_config.color_hints == "compressed"
        )

    def callback(
        self,
        color_data: Union[Image, CompressedImage],
        depth_data: Optional[Union[Image, CompressedImage]] = None,
        camera_info: Optional[CameraInfo] = None,
    ) -> None:
        """
        Process synchronized camera data.

        This callback handles incoming color, depth, and camera info messages. It
        converts the data to OpenCV format and stores it for later use.

        TODO make this generic. handle the encoding and order properly. For standard and
        compressed images. this might also depend on the fix of image_transport_plugins
        being published as a package. Startpoint can be found at the bottom of this
        method.

        :param color_data: Color image message
        :param depth_data: Depth image message
        :param camera_info: Camera calibration message
        """
        with self.lock:
            if self.rk_logger.isEnabledFor(logging.DEBUG):
                self.rk_logger.debug("Received data:")

                color_time = Time(
                    seconds=color_data.header.stamp.sec,
                    nanoseconds=color_data.header.stamp.nanosec,
                )

                if depth_data is not None:
                    depth_time = Time(
                        seconds=depth_data.header.stamp.sec,
                        nanoseconds=depth_data.header.stamp.nanosec,
                    )
                    self.rk_logger.debug(
                        f"  Color time - Depth time: {(color_time - depth_time).nanoseconds / 1e9:.6f}"
                    )

                if camera_info is not None:
                    camera_info_time = Time(
                        seconds=camera_info.header.stamp.sec,
                        nanoseconds=camera_info.header.stamp.nanosec,
                    )
                    self.rk_logger.debug(
                        f"  Color time - Camera Info time: {(color_time - camera_info_time).nanoseconds / 1e9:.6f}"
                    )

            if self.compressed_color_configured():
                color_arr = np.frombuffer(color_data.data, np.uint8)
                self.color = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
            else:
                self.color = self.bridge.imgmsg_to_cv2(color_data, "bgr8")

            self.timestamp = color_data.header.stamp

            if self.compressed_depth_configured():
                if depth_data is None:
                    raise CameraDataMissing(
                        data_name="Depth data",
                        context="compressed depth conversion",
                    )
                self.depth = depth_convert_workaround(depth_data)
            else:
                if depth_data is None:
                    raise CameraDataMissing(
                        data_name="Depth data",
                        context="uncompressed depth conversion",
                    )
                self.depth = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")

            self.camera_info = camera_info

            self._has_new_data = True

    def set_data(self, cas: CAS, world: World) -> None:
        if not self.has_new_data():
            return

        with self.lock:
            if self.camera_config.hi_res_mode:
                self.color = self.color[0:960, 0:1280]

            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
            width = self.camera_info.width
            height = self.camera_info.height
            if self.camera_config.hi_res_mode:
                height = 960

            fx = self.camera_info.k[0]
            cx = self.camera_info.k[2]
            fy = self.camera_info.k[4]
            cy = self.camera_info.k[5]
            self.camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

            self.color_to_depth_ratio = self.camera_config.color_to_depth_ratio

            cas.set(CASViews.COLOR_IMAGE, self.color)
            cas.set(CASViews.DEPTH_IMAGE, self.depth)
            cas.set(CASViews.CAMERA_INFO, self.camera_info)
            cas.set(CASViews.CAMERA_INTRINSIC, self.camera_intrinsic)
            cas.set(CASViews.COLOR_TO_DEPTH_RATIO, self.color_to_depth_ratio)

            self.place_camera(cas, world, self.timestamp)

            self._has_new_data = False
