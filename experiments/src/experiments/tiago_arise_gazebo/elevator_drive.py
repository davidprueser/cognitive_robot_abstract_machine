"""
The elevator of the TIAGo ARISE Gazebo simulator, which reports its cabin's height as
joint states.

The real elevator reports nothing, so this exists for the simulator only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from sensor_msgs.msg import JointState

from semantic_digital_twin.adapters.ros.input_synchronization import (
    TopicInputSynchronizer,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator

# %% topics


class SimulatorTopic(StrEnum):
    """
    Topics of the TIAGo ARISE Gazebo simulator.
    """

    ELEVATOR_JOINT_STATES = "/elevator/joint_states"


# %% elevator drive


@dataclass
class ElevatorDrive(TopicInputSynchronizer[JointState]):
    """
    Writes the position of the simulated elevator's drive into the world, so that the
    cabin stands where the simulator's cabin stands.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator whose cabin is moved.
    """

    joint_name: str = field(kw_only=True)
    """
    Name of the drive joint in the joint states.
    """

    position_tolerance: float = field(default=0.001, kw_only=True)
    """
    How far the received position may differ from the cabin's before the cabin is moved,
    so that an elevator standing still causes no world changes.
    """

    position_offset: float = field(init=False)
    """
    The drive position minus the joint position the simulator reports, read from the
    world when the drive is created.

    .. note:: The simulator's joint is 0 where the cabin serves the lowest floor.
    """

    def __post_init__(self) -> None:
        """
        Read :attr:`position_offset` from the world, so that applying a message computes
        nothing in it.
        """
        self.position_offset = self.elevator.drive_position_for_floor(
            self.elevator.floors[0]
        )
        super().__post_init__()

    def take_message(self) -> JointState | None:
        """
        :return: The latest joint states, unless they name no position of the drive
            joint or one within :attr:`position_tolerance` of the cabin's.
        """
        message = self.latest_message
        if message is None or self.joint_name not in message.name:
            return None
        if (
            abs(self.drive_position(message) - self.elevator.movable_joint.position)
            <= self.position_tolerance
        ):
            return None
        return message

    def apply_message(self, message: JointState) -> None:
        self.world.state[self.elevator.movable_joint.raw_dof.id].position = (
            self.drive_position(message)
        )

    def drive_position(self, message: JointState) -> float:
        """
        :return: The drive position ``message`` reports for the drive joint.
        """
        return (
            message.position[message.name.index(self.joint_name)] + self.position_offset
        )
