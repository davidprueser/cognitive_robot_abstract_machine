from __future__ import annotations
from dataclasses import dataclass


from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.io.storage_reader_interface import StorageReaderInterface
from robokudo.descriptors.camera_configs.components import WorldDescriptorComponent


@dataclass(kw_only=True)
class MongoCameraConfig(
    BaseCameraConfig[StorageReaderInterface], WorldDescriptorComponent
):
    """
    Configuration class for MongoDB-based camera data playback.

    This class defines the configuration parameters for reading camera data from a
    MongoDB database, typically used for offline processing or testing with stored
    scene data.

    .. note::
        This configuration is used to read previously stored camera data from a
        MongoDB database, allowing for replay and analysis of recorded scenes.
    """

    loop: bool = True
    """
    Flag to enable looping over database entries.
    """

    db_name: str = "rk_scenes"
    """
    Name of the MongoDB database to read from.
    """

    restore_annotations: bool = False
    """
    Whether to restore stored CAS annotations during playback.
    """
