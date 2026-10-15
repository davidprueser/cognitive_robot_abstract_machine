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
class OrbbecCameraConfig(
    BaseCameraConfig[KinectCameraInterface],
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """
    Configuration class for the OrbBec Astra camera.
    """

    topic_depth: str = "/camera/depth/image_raw"

    topic_color: str = "/camera/color/image_raw/compressed"

    topic_camera_info: str = "/camera/color/camera_info"

    tf_from: str = "camera_color_optical_frame"

    tf_to: str = "map"

    depth_hints: str = "raw"

    filterBlurredImages: bool = False
