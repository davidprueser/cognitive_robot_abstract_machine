from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithDiffDriveRobot
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago, TiagoJoint
from semantic_digital_twin.world_description.connections import DifferentialDrive, Connection6DoF


@dataclass
class WorldWithTiagoConfig(WorldWithDiffDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Tiago, init=False)
    translation_velocity_limit: float = field(kw_only=True, default=0.5)
    rotation_velocity_limit: float = field(kw_only=True, default=0.6)


@dataclass
class TiagoStandaloneInterface(RobotInterfaceConfig):
    """
    Simulates Tiago joints and drive without talking to hardware.
    """

    def setup(self):
        self.register_controlled_joints(
            [
                # TiagoJoint.TORSO_LIFT,
                # TiagoJoint.HEAD_1,
                # TiagoJoint.HEAD_2,
                # TiagoJoint.LEFT_ARM_1,
                # TiagoJoint.LEFT_ARM_2,
                # TiagoJoint.LEFT_ARM_3,
                # TiagoJoint.LEFT_ARM_4,
                # TiagoJoint.LEFT_ARM_5,
                # TiagoJoint.LEFT_ARM_6,
                # TiagoJoint.LEFT_ARM_7,
                # TiagoJoint.LEFT_GRIPPER_FINGER,
                # TiagoJoint.RIGHT_ARM_1,
                # TiagoJoint.RIGHT_ARM_2,
                # TiagoJoint.RIGHT_ARM_3,
                # TiagoJoint.RIGHT_ARM_4,
                # TiagoJoint.RIGHT_ARM_5,
                # TiagoJoint.RIGHT_ARM_6,
                # TiagoJoint.RIGHT_ARM_7,
                # TiagoJoint.RIGHT_GRIPPER_FINGER,
                self.world.get_connections_by_type(DifferentialDrive)[0].name,
            ]
        )


@dataclass
class TiagoVelocityInterface(RobotInterfaceConfig):
    """
    Commands Tiago through controller-manager-discovered velocity interfaces.
    """

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        self.sync_robot_parts()

        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.add_base_cmd_velocity(cmd_vel_topic="/cmd_vel", joint=diff_drive)


    #
    # def setup(self):
    #     self.discover_interfaces_from_controller_manager()
    #     self.sync_robot_parts()
    #     diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
    #     self.add_base_cmd_velocity(cmd_vel_topic="/cmd_vel", joint=diff_drive)
