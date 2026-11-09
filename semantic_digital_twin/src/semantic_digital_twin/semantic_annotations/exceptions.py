from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Level

# %% ground to drive on


@dataclass
class NoFloorOnLevelError(UsageError):
    """
    Raised when a level holds no floor, so that it offers no ground to drive on.
    """

    level: Level
    """
    The level that was asked for its ground.
    """

    def error_message(self) -> str:
        return f"'{self.level.root.name}' holds no floor at the height of its region."

    def suggest_correction(self) -> str:
        return (
            "spawn the storey's floors with their tops at the bottom of the level's "
            "region, within its footprint."
        )


@dataclass
class NoSupportingSurfaceError(UsageError):
    """
    Raised when an annotation's geometry offers no surface anything could be supported
    on.
    """

    annotation: HasSupportingSurface
    """
    The annotation that was asked for its supporting surface.
    """

    def error_message(self) -> str:
        return (
            f"'{self.annotation.root.name}' has no supporting surface and none could "
            f"be derived from its geometry."
        )

    def suggest_correction(self) -> str:
        return (
            "attach a supporting surface region to the annotation, or give its root "
            "body geometry with an upward facing face."
        )
