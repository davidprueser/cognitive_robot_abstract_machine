from __future__ import annotations
from dataclasses import dataclass


from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.io.ros_camera_without_depth_interface import (
    ROSCameraWithoutDepthInterface,
)
from robokudo.descriptors.camera_configs.components import (
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)


@dataclass(kw_only=True)
class HsrHandCameraConfig(
    BaseCameraConfig[ROSCameraWithoutDepthInterface],
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """
    Camera config for the camera-in-hand on the Toyota HSR robot.
    """

    rotate_image: str = "90_ccw"
    """Rotate the image cw = clockwise, ccw = counter-clockwise. Possible values: None, 90_{cw, ccw}, 180_{cw, ccw}."""

    topic_color: str = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"

    topic_camera_info: str = "hsrb/head_rgbd_sensor/rgb/camera_info"

    tf_from: str = "hand_camera_frame"

    tf_to: str = "map"
