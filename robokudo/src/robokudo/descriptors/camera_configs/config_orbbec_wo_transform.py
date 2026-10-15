from __future__ import annotations
from dataclasses import dataclass


from robokudo.descriptors.camera_configs.config_orbbec import OrbbecCameraConfig


@dataclass(kw_only=True)
class OrbbecWoTfCameraConfig(OrbbecCameraConfig):
    """
    Configuration class for the OrbBec Astra camera without transform lookup.
    """

    lookup_viewpoint: bool = False
