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
class RealsenseCameraConfig(
    BaseCameraConfig[KinectCameraInterface],
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """
    Configuration class for Intel RealSense cameras.

    This class defines the configuration parameters for RealSense cameras,
    particularly tested with the D435 model. It assumes the use of aligned depth
    data and requires the RealSense ROS driver to be running.

    .. note::
        Requires the RealSense ROS driver to be running with aligned depth:
        roslaunch realsense2_camera rs_aligned_depth.launch
    """

    topic_depth: str = "/camera/aligned_depth_to_color/image_raw/compressedDepth"

    topic_color: str = "/camera/color/image_raw/compressed"

    topic_camera_info: str = "/camera/color/camera_info"

    tf_from: str = "camera_color_optical_frame"

    tf_to: str = "map"

    filterBlurredImages: bool = False
