from __future__ import annotations

from dataclasses import dataclass

import pytest
from krrood.exceptions import UnboundGenericParameter

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.base_camera_config import (
    BaseCameraConfig,
    CameraInterfaceType,
)
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.io.storage_reader_interface import StorageReaderInterface

# %% configurations bound to interfaces the tests provide


@dataclass
class RecordingInterface:
    """
    A camera interface that only remembers the configuration it was created for.
    """

    camera_config: BaseCameraConfig
    """
    The configuration the interface reads.
    """


@dataclass(kw_only=True)
class RecordingCameraConfig(BaseCameraConfig[RecordingInterface]):
    """
    A camera configuration whose interface is :class:`RecordingInterface`.
    """

    topic: str = "/camera/color"
    """
    A setting of the camera, to tell configurations apart.
    """


@dataclass(kw_only=True)
class UnboundCameraConfig(BaseCameraConfig):
    """
    A camera configuration that names no interface.
    """


# %% interface binding


def test_config_creates_its_bound_interface_for_itself() -> None:
    config = RecordingCameraConfig()

    interface = config.create_interface()

    assert isinstance(interface, RecordingInterface)
    assert interface.camera_config is config


def test_config_without_interface_cannot_create_one() -> None:
    with pytest.raises(UnboundGenericParameter):
        UnboundCameraConfig().create_interface()


def test_storage_playback_reads_through_the_storage_reader() -> None:
    assert (
        MongoCameraConfig.get_type_of_generic_parameter(CameraInterfaceType)
        is StorageReaderInterface
    )


# %% collection reader


def test_collection_reader_creates_an_interface_per_camera_in_setup() -> None:
    main_camera = RecordingCameraConfig(topic="/main")
    second_camera = RecordingCameraConfig(topic="/second")
    reader = CollectionReaderAnnotator(camera_config=main_camera)
    reader.add_collection_reader(second_camera)

    reader.setup()

    assert [interface.camera_config for interface in reader.camera_interfaces] == [
        main_camera,
        second_camera,
    ]


def test_collection_reader_creates_no_interface_before_setup() -> None:
    reader = CollectionReaderAnnotator(camera_config=RecordingCameraConfig())

    assert reader.camera_interfaces == []
