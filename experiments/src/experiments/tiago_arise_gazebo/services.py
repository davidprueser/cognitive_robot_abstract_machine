"""
The services of the TIAGo ARISE Gazebo simulator.
"""

from __future__ import annotations

from enum import StrEnum

# %% services


class SimulatorService(StrEnum):
    """
    Services of the TIAGo ARISE Gazebo simulator.
    """

    ELEVATOR_OPERATOR_PARAMETERS = "/elevator_operator/set_parameters"
    """
    Sets the parameters of the simulator's elevator operator.
    """

    SET_ENTITY_POSE = "/set_entity_pose"
    """
    Moves an entity of the simulated world, such as the robot, to a pose.
    """
