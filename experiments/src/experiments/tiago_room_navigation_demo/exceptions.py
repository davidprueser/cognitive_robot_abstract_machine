from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% planning the way to a room


@dataclass
class RoomOnNoLevelError(DataclassException):
    """
    Raised when the room a robot is sent to lies on no level of its building.
    """

    room_name: str
    """
    The name of the room the robot was sent to.
    """

    def error_message(self) -> str:
        return f"The room {self.room_name} lies on no level."

    def suggest_correction(self) -> str:
        return "Annotate the storey the room's floor lies on as a level."


@dataclass
class NoElevatorError(DataclassException):
    """
    Raised when a robot has to change levels to reach a room, but its world holds no
    elevator.
    """

    room_name: str
    """
    The name of the room on another level the robot was sent to.
    """

    def error_message(self) -> str:
        return (
            f"The room {self.room_name} lies on another level than the robot, and no "
            f"elevator leads there."
        )

    def suggest_correction(self) -> str:
        return (
            "Add an elevator to the building, or send the robot to a room on its level."
        )


# %% placing the robot


@dataclass
class RobotNotPlacedError(DataclassException):
    """
    Raised when a robot moved into the room it starts in does not show up there.
    """

    room_name: str
    """
    The name of the room the robot was moved into.
    """

    def error_message(self) -> str:
        return f"The robot does not show up in the room {self.room_name}."

    def suggest_correction(self) -> str:
        return (
            "Check that the simulator moved the robot and that its odometry reaches "
            "the world."
        )


# %% checking the outcome


@dataclass
class TargetRoomNotReachedError(DataclassException):
    """
    Raised when a robot finished its plan without standing in the room it was sent to.
    """

    room_name: str
    """
    The name of the room the robot was sent to.
    """

    def error_message(self) -> str:
        return f"The robot does not stand on the floor of the room {self.room_name}."

    def suggest_correction(self) -> str:
        return "Check where the plan left the robot and whether its path ended early."
