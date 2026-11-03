"""
Moving an entity of the TIAGo ARISE Gazebo simulator's world, such as the robot, to a
pose at once.

The real world cannot be moved like this, so this exists for the simulator only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from experiments.tiago_arise_gazebo.exceptions import (
    EntityTeleportRejectedError,
    EntityTeleportUnavailableError,
)
from experiments.tiago_arise_gazebo.services import SimulatorService
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% teleporting


@dataclass
class SimulatedEntityTeleport:
    """
    Moves a model of the simulated world to a pose in the world frame, without it
    travelling there.
    """

    node: Node = field(kw_only=True)
    """
    The node the simulator is asked through.
    """

    service_name: str = field(default=SimulatorService.SET_ENTITY_POSE, kw_only=True)
    """
    The service that sets the pose of an entity.
    """

    timeout: float = field(default=10.0, kw_only=True)
    """
    Seconds to wait for the simulator to offer the service and to answer.
    """

    def teleport(self, model_name: str, pose: Pose) -> None:
        """
        :param model_name: The name of the model in the simulated world.
        :param pose: Where the model's frame is put, in the frame of the simulated
            world.
        :raises EntityTeleportUnavailableError: If the simulator offers no way to move
            the model within :attr:`timeout`.
        :raises EntityTeleportRejectedError: If the simulator does not move the model.
        """
        client = self.node.create_client(SetEntityPose, self.service_name)
        if not client.wait_for_service(timeout_sec=self.timeout):
            self.node.destroy_client(client)
            raise EntityTeleportUnavailableError(self.service_name)
        response = client.call(
            SetEntityPose.Request(
                entity=Entity(name=model_name, type=Entity.MODEL),
                pose=PoseToRos2Converter.convert(pose),
            ),
            timeout_sec=self.timeout,
        )
        self.node.destroy_client(client)
        if response is None or not response.success:
            raise EntityTeleportRejectedError(model_name)
