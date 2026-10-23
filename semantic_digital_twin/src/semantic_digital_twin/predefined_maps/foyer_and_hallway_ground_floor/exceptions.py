from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types import Point2

# %% room outline


@dataclass
class NonRectilinearOutlineError(UsageError):
    """
    Raised when a room outline has an edge that is not parallel to the x- or y-axis.
    """

    start: Point2
    """
    The first vertex of the offending edge.
    """

    end: Point2
    """
    The second vertex of the offending edge.
    """

    def error_message(self) -> str:
        return (
            f"The outline edge from {self.start.to_np()} to {self.end.to_np()} is "
            f"neither parallel to the x-axis nor to the y-axis."
        )

    def suggest_correction(self) -> str:
        return (
            "Give every pair of consecutive vertices either the same x or the same y."
        )


@dataclass
class DoorwayOffOutlineError(UsageError):
    """
    Raised when the jambs of a doorway do not both lie on the same edge of a room
    outline.
    """

    start: Point2
    """
    The first jamb of the doorway.
    """

    end: Point2
    """
    The second jamb of the doorway.
    """

    def error_message(self) -> str:
        return (
            f"The doorway from {self.start.to_np()} to {self.end.to_np()} does not lie "
            f"on a single edge of the room outline."
        )

    def suggest_correction(self) -> str:
        return "Place both jambs of the doorway on the same outline edge."


@dataclass
class OpenEdgeOffOutlineError(UsageError):
    """
    Raised when an edge meant to be left without a wall is not a whole edge of the room
    outline.
    """

    start: Point2
    """
    The first vertex of the open edge.
    """

    end: Point2
    """
    The second vertex of the open edge.
    """

    def error_message(self) -> str:
        return (
            f"The open edge from {self.start.to_np()} to {self.end.to_np()} is not an "
            f"edge of the room outline."
        )

    def suggest_correction(self) -> str:
        return "Give the two consecutive outline vertices the edge runs between."


@dataclass
class DoorwayOnOpenEdgeError(UsageError):
    """
    Raised when a doorway is placed on a room outline edge that has no wall to cut it
    into.
    """

    start: Point2
    """
    The first jamb of the doorway.
    """

    end: Point2
    """
    The second jamb of the doorway.
    """

    def error_message(self) -> str:
        return (
            f"The doorway from {self.start.to_np()} to {self.end.to_np()} lies on an "
            f"outline edge that is left without a wall."
        )

    def suggest_correction(self) -> str:
        return "Remove the doorway or the open edge it lies on."
