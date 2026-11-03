import threading

import pytest
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from experiments.tiago_arise_gazebo.elevator_operator import (
    ElevatorOperatorParameter,
    SimulatedElevatorOperator,
)
from experiments.tiago_arise_gazebo.exceptions import (
    DestinationFloorRejectedError,
    ElevatorOperatorUnavailableError,
)

# %% fixtures


@pytest.fixture
def operator_node(rclpy_node) -> Node:
    """
    A node standing in for the simulator's elevator operator, which knows three floors.
    """
    node = rclpy.create_node("elevator_operator")
    node.declare_parameter(ElevatorOperatorParameter.DESTINATION_FLOOR, 1)
    node.add_on_set_parameters_callback(
        lambda parameters: SetParametersResult(
            successful=all(parameter.value <= 2 for parameter in parameters)
        )
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    thread.join()
    node.destroy_node()


# %% telling the operator the destination floor


def test_the_operator_takes_the_destination_floor(rclpy_node, operator_node):
    SimulatedElevatorOperator(node=rclpy_node).set_destination_floor(2)

    assert (
        operator_node.get_parameter(ElevatorOperatorParameter.DESTINATION_FLOOR).value
        == 2
    )


def test_a_floor_the_operator_does_not_take_is_reported(rclpy_node, operator_node):
    with pytest.raises(DestinationFloorRejectedError):
        SimulatedElevatorOperator(node=rclpy_node).set_destination_floor(3)


def test_a_missing_operator_is_reported(rclpy_node):
    with pytest.raises(ElevatorOperatorUnavailableError):
        SimulatedElevatorOperator(
            node=rclpy_node,
            parameter_service_name="/no_elevator_operator/set_parameters",
            timeout=0.2,
        ).set_destination_floor(1)
