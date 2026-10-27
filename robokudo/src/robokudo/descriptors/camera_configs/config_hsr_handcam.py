from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import (
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)
from robokudo.types.cv import ImageRotation


@dataclass(slots=True)
class HsrHandCameraConfig(
    BaseCameraConfig,
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """
    Camera config for the camera-in-hand on the Toyota HSR robot.
    """

    registry_name: ClassVar[str] = "hsr_handcam"

    image_rotation: ImageRotation | None = ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE
    """
    How to rotate each image; None keeps it as it is.
    """

    topic_color: str = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"

    topic_camera_info: str = "hsrb/head_rgbd_sensor/rgb/camera_info"

    tf_from: str = "hand_camera_frame"

    tf_to: str = "map"
