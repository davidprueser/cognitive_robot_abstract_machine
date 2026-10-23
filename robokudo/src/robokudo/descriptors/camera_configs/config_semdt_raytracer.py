from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import RayTracedSceneComponent


@dataclass(slots=True)
class SemDTRayTracerCameraConfig(BaseCameraConfig, RayTracedSceneComponent):
    """
    Configuration for a simulated RGB-D camera backed by SemDT's RayTracer.
    """

    registry_name: ClassVar[str] = "semdt_raytracer"

    interface_type: str = "SemDTRayTracer"

    camera_x: float = -1.20
    camera_y: float = 0.40
    camera_z: float = 1.05
    camera_roll: float = -2.2689280275926285
    camera_pitch: float = 0.0
    camera_yaw: float = -0.2707963267948965
    """
    Camera optical-frame pose (x right, y down, z forward) in world frame (xyz + rpy).
    """
