from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import RayTracedSceneComponent


@dataclass(slots=True)
class SemDTRayTracerRobotCameraConfig(BaseCameraConfig, RayTracedSceneComponent):
    """
    Configuration for a simulated RGB-D camera that renders a world descriptor's scene
    from where the world perception perceives into holds a robot's camera.
    """

    registry_name: ClassVar[str] = "semdt_raytracer_robot_camera"

    interface_type: str = "SemDTRayTracerRobotCamera"

    camera_frame: str = "head_mount_kinect_rgb_optical_frame"
    """
    Name of the robot's camera optical frame, in the world perception perceives into and
    in the rendered scene.
    """
