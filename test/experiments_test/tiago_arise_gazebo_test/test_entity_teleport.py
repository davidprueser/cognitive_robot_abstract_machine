import threading

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose

from experiments.tiago_arise_gazebo.entity_teleport import SimulatedEntityTeleport
from experiments.tiago_arise_gazebo.exceptions import (
    EntityTeleportRejectedError,
    EntityTeleportUnavailableError,
)
from experiments.tiago_arise_gazebo.services import SimulatorService
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% fixtures


@pytest.fixture
def received_requests() -> list[SetEntityPose.Request]:
    """
    The requests the stand-in simulator received, in order.
    """
    return []


@pytest.fixture
def simulator_node(rclpy_node, received_requests) -> Node:
    """
    A node standing in for the simulator, whose world holds one model, ``tiago``.
    """
    node = rclpy.create_node("simulator")

    def set_entity_pose(
        request: SetEntityPose.Request, response: SetEntityPose.Response
    ) -> SetEntityPose.Response:
        received_requests.append(request)
        response.success = request.entity.name == "tiago"
        return response

    node.create_service(
        SetEntityPose, SimulatorService.SET_ENTITY_POSE, set_entity_pose
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    thread.join()
    node.destroy_node()


# %% teleporting a model


def test_the_simulator_is_asked_to_put_the_model_at_the_pose(
    rclpy_node, simulator_node, received_requests
):
    pose = Pose.from_xyz_rpy(x=1.0, y=-2.0, z=3.45, yaw=0.5)

    SimulatedEntityTeleport(node=rclpy_node).teleport("tiago", pose)

    [request] = received_requests
    assert request.entity.name == "tiago"
    assert [
        request.pose.position.x,
        request.pose.position.y,
        request.pose.position.z,
    ] == pytest.approx(pose.position.to_np()[:3].tolist())
    assert [
        request.pose.orientation.x,
        request.pose.orientation.y,
        request.pose.orientation.z,
        request.pose.orientation.w,
    ] == pytest.approx(pose.rotation_matrix.quaternion.to_np().tolist())


def test_a_model_the_simulator_does_not_move_is_reported(rclpy_node, simulator_node):
    with pytest.raises(EntityTeleportRejectedError):
        SimulatedEntityTeleport(node=rclpy_node).teleport("no_such_model", Pose())


def test_a_missing_simulator_is_reported(rclpy_node):
    with pytest.raises(EntityTeleportUnavailableError):
        SimulatedEntityTeleport(
            node=rclpy_node, service_name="/no_simulator/set_pose", timeout=0.2
        ).teleport("tiago", Pose())
