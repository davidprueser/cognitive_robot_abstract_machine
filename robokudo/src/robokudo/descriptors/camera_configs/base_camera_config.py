from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from robokudo.io.camera_interface import CameraInterface

CameraInterfaceType = TypeVar("CameraInterfaceType", bound="CameraInterface")

# %% camera configurations


@dataclass(kw_only=True)
class BaseCameraConfig(Generic[CameraInterfaceType], SubClassSafeGeneric):
    """
    Configuration of a camera, bound to the interface that reads its data.
    """

    def create_interface(self) -> CameraInterfaceType:
        """
        :return: A new interface reading the data of this camera.
        """
        interface_type = self.get_type_of_generic_parameter(CameraInterfaceType)
        return interface_type(self)
