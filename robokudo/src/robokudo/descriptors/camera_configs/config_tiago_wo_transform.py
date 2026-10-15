from __future__ import annotations
from dataclasses import dataclass


from robokudo.descriptors.camera_configs.config_tiago import TiagoCameraConfig


@dataclass(kw_only=True)
class TiagoWoTfCameraConfig(TiagoCameraConfig):
    """
    Configuration class for the TIAGo robot's Xtion camera without transform lookup.

    This class defines the configuration parameters for the Xtion RGB-D camera
    mounted on the TIAGo robot, similar to config_tiago.py but with viewpoint
    lookup disabled. It includes settings for camera interface and topic names.

    .. note::
        This configuration is identical to config_tiago.py except that
        lookup_viewpoint is set to False by default.
    """

    lookup_viewpoint: bool = False
