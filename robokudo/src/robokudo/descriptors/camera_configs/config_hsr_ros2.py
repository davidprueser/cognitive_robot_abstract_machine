from __future__ import annotations
from dataclasses import dataclass


from robokudo.descriptors.camera_configs.components import (
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.io.camera_interface import KinectCameraInterface


@dataclass(kw_only=True)
class HsrRos2CameraConfig(
    BaseCameraConfig[KinectCameraInterface],
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """
    Camera config for the HSR robot.
    """

    topic_depth: str = "/head_rgbd_sensor/depth_registered/image/compressedDepth"

    topic_color: str = "/head_rgbd_sensor/rgb/image_rect_color/compressed"

    topic_camera_info: str = "/head_rgbd_sensor/rgb/camera_info"

    tf_from: str = "head_rgbd_sensor_rgb_optical_frame"

    tf_to: str = "map"
