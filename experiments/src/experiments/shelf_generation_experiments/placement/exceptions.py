from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from experiments.shelf_generation_experiments.utils import ObjectType


# %% placement
@dataclass
class NoShelfPlacementError(DataclassException):
    """
    Raised when no layer of a shelf has room for an object, according to the shelf
    model.
    """

    shelf_name: str
    """
    Name of the shelf corpus that was asked.
    """

    object_type: ObjectType
    """
    Type of the object that found no room.
    """

    def error_message(self) -> str:
        return (
            f"No layer of shelf {self.shelf_name!r} has room for a "
            f"{self.object_type.value} object."
        )

    def suggest_correction(self) -> str:
        return "Place a smaller object, or a shelf whose layers leave more free space."
