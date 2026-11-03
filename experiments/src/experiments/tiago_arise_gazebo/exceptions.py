from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% elevator operator


@dataclass
class ElevatorOperatorUnavailableError(DataclassException):
    """
    Raised when the simulator's elevator operator offers no way to set its parameters.
    """

    parameter_service_name: str
    """
    The service the operator's parameters were to be set through.
    """

    def error_message(self) -> str:
        return f"No elevator operator answers on {self.parameter_service_name}."

    def suggest_correction(self) -> str:
        return (
            "Start the simulator with its elevator operator, or tell the demonstration "
            "that a person operates the elevator."
        )


@dataclass
class DestinationFloorRejectedError(DataclassException):
    """
    Raised when the simulator's elevator operator does not take a destination floor.
    """

    floor_index: int
    """
    The floor the operator was told, counted from the lowest floor.
    """

    def error_message(self) -> str:
        return f"The elevator operator did not take floor {self.floor_index}."

    def suggest_correction(self) -> str:
        return "Check that the floor is one of the floors the operator knows."
