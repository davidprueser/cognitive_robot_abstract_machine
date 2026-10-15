"""
Annotator for writing a configured static camera transform into the CAS.

Use this in scenarios where you do not have a world to camera transform published or
when you develop the perception system in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from py_trees.common import Status
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.camera_interface import CameraInterface


@dataclass(kw_only=True)
class StaticCameraTransformAnnotator(BaseAnnotator):
    """
    Write a fixed camera-to-world transform to the current CAS.
    """

    world_frame: str = "map"
    """
    Reference frame used for the camera-to-world transform.
    """

    camera_frame: str = "camera"
    """
    Camera frame used as the transform child frame.
    """

    world_T_camera: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Camera pose relative to the configured world frame.
    """

    def compute(self) -> Status:

        cas = self.get_cas()

        CameraInterface.store_camera_to_world_transform_in_cas(
            cas=cas,
            world_frame=self.world_frame,
            camera_frame=self.camera_frame,
            world_T_camera=self.world_T_camera,
        )

        return Status.SUCCESS
