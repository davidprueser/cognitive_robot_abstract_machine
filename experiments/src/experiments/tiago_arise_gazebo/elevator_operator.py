"""
The elevator operator of the TIAGo ARISE Gazebo simulator, which brings the cabin to a
robot waiting for it and rides a boarded robot to the floor it has been told.

The real elevator is operated by a person, so this exists for the simulator only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter

from experiments.tiago_arise_gazebo.services import SimulatorService
from experiments.tiago_arise_gazebo.exceptions import (
    DestinationFloorRejectedError,
    ElevatorOperatorUnavailableError,
)

# %% parameters


class ElevatorOperatorParameter(StrEnum):
    """
    Parameters of the simulator's elevator operator.
    """

    DESTINATION_FLOOR = "destination_floor"
    """
    The floor a boarded robot is ridden to, counted from the lowest floor.
    """


# %% elevator operator


@dataclass
class SimulatedElevatorOperator:
    """
    Tells the simulator's elevator operator which floor to ride a boarded robot to.
    """

    node: Node = field(kw_only=True)
    """
    The node the operator's parameters are set through.
    """

    parameter_service_name: str = field(
        default=SimulatorService.ELEVATOR_OPERATOR_PARAMETERS, kw_only=True
    )
    """
    The service that sets the operator's parameters.
    """

    timeout: float = field(default=10.0, kw_only=True)
    """
    Seconds to wait for the operator to offer its parameters and to answer.
    """

    def set_destination_floor(self, floor_index: int) -> None:
        """
        :param floor_index: The floor a boarded robot is ridden to, counted from the
            lowest floor.
        :raises ElevatorOperatorUnavailableError: If the operator offers no way to set
            its parameters within :attr:`timeout`.
        :raises DestinationFloorRejectedError: If the operator does not take the floor.
        """
        client = self.node.create_client(SetParameters, self.parameter_service_name)
        if not client.wait_for_service(timeout_sec=self.timeout):
            self.node.destroy_client(client)
            raise ElevatorOperatorUnavailableError(self.parameter_service_name)
        response = client.call(
            SetParameters.Request(
                parameters=[
                    Parameter(
                        ElevatorOperatorParameter.DESTINATION_FLOOR, value=floor_index
                    ).to_parameter_msg()
                ]
            ),
            timeout_sec=self.timeout,
        )
        self.node.destroy_client(client)
        if response is None or not response.results[0].successful:
            raise DestinationFloorRejectedError(floor_index)
