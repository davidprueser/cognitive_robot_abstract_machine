from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from experiments.shelf_generation_experiments.utils import ObjectType


# %% demo setup
@dataclass
class NoFittingObjectError(DataclassException):
    """
    Raised when none of the objects on offer fits any layer of the generated shelf.
    """

    object_type: ObjectType
    """
    The type the objects on offer share.
    """

    shortest_height: float
    """
    Height, in metres, of the shortest object on offer.
    """

    layer_heights: list[float]
    """
    The room above every layer's slab, in metres.
    """

    def error_message(self) -> str:
        rooms = ", ".join(f"{height:.3f}" for height in self.layer_heights)
        return (
            f"No {self.object_type.value} fits the shelf: the shortest is "
            f"{self.shortest_height:.3f} m tall and the layers offer {rooms} m."
        )

    def suggest_correction(self) -> str:
        return "Generate a shelf with fewer layers, which leaves more room above each."


@dataclass
class NoStandingMeshError(DataclassException):
    """
    Raised when no placeable object type has a mesh that stands upright.
    """

    def error_message(self) -> str:
        return (
            "No object type both themes shelves and has a mesh, among the downloaded "
            "scenes, that is taller than it is wide."
        )

    def suggest_correction(self) -> str:
        return "Download more sage10k scenes into the scenes root."


# %% navigation
@dataclass
class UnreachableShelfError(DataclassException):
    """
    Raised when the free space of the floor holds no route from the robot to the shelf.
    """

    walking_distance: float
    """
    Straight-line distance, in metres, from the robot to where it has to stand.
    """

    floor_occupant_names: list[str]
    """
    Names of what stands on the floor and divides its free space.
    """

    def error_message(self) -> str:
        return (
            f"No route across the floor reaches the shelf {self.walking_distance:.3f} m "
            f"away; {', '.join(self.floor_occupant_names)} leave no way through."
        )

    def suggest_correction(self) -> str:
        return "Move what stands between the robot and the shelf, or move the shelf."
