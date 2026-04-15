import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    HasLeftRightArm,
    HasNeck,
    HasTorso,
    AbstractRobot,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    ParallelGripper,
    Arm,
    Torso,
    FieldOfView,
    Camera,
    HumanoidGripper,
    Neck,
    MobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class Armar6(AbstractRobot, HasLeftRightArm, HasTorso, HasNeck, HasMobileBase):
    """
    Representation of the Armar6 robot, see https://h2t.iar.kit.edu/397.php#gallery-1
    """

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Body:
        return world.get_body_by_name("base_footprint")

    def _setup_arm_semantic_annotations(self):
        world = self._world
        # Create left arm
        left_gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root_name="l_gripper_l_finger_link",
            tip_name="l_gripper_l_finger_tip_link",
            world=world,
        )
        left_gripper_index = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_finger", prefix=self.name.name),
            root_name="l_gripper_r_finger_link",
            tip_name="l_gripper_r_finger_tip_link",
            world=world,
        )
        left_gripper_middle = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_middle", prefix=self.name.name),
            root_name="l_gripper_r_middle_link",
            tip_name="l_gripper_r_middle_tip_link",
            world=world,
        )

        left_gripper_ring = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_ring", prefix=self.name.name),
            root_name="l_gripper_r_ring_link",
            tip_name="l_gripper_r_ring_tip_link",
            world=world,
        )

        left_gripper_pinky = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_pinky", prefix=self.name.name),
            root_name="l_gripper_r_pinky_link",
            tip_name="l_gripper_r_pinky_tip_link",
            world=world,
        )

        left_gripper = HumanoidGripper.create_and_add_to_world(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root_name="l_gripper_palm_link",
            tool_frame_name="l_gripper_tool_frame",
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            thumb=left_gripper_thumb,
            fingers=[
                left_gripper_index,
                left_gripper_middle,
                left_gripper_ring,
                left_gripper_pinky,
            ],
            world=world,
        )

        left_arm = Arm.create_and_add_to_world(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root_name="torso_lift_link",
            tip_name="l_wrist_roll_link",
            manipulator=left_gripper,
            world=world,
        )
        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root_name="r_gripper_l_finger_link",
            tip_name="r_gripper_l_finger_tip_link",
            world=world,
        )
        right_gripper_index = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_finger", prefix=self.name.name),
            root_name="r_gripper_r_finger_link",
            tip_name="r_gripper_r_finger_tip_link",
            world=world,
        )
        right_gripper_middle = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_middle", prefix=self.name.name),
            root_name="r_gripper_r_middle_link",
            tip_name="r_gripper_r_middle_tip_link",
            world=world,
        )

        right_gripper_ring = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_ring", prefix=self.name.name),
            root_name="r_gripper_r_ring_link",
            tip_name="r_gripper_r_ring_tip_link",
            world=world,
        )

        right_gripper_pinky = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_pinky", prefix=self.name.name),
            root_name="r_gripper_r_pinky_link",
            tip_name="r_gripper_r_pinky_tip_link",
            world=world,
        )

        right_gripper = HumanoidGripper.create_and_add_to_world(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root_name="r_gripper_palm_link",
            tool_frame_name="r_gripper_tool_frame",
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            thumb=right_gripper_thumb,
            fingers=[
                right_gripper_index,
                right_gripper_middle,
                right_gripper_ring,
                right_gripper_pinky,
            ],
            world=world,
        )

        right_arm = Arm.create_and_add_to_world(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root_name="torso_lift_link",
            tip_name="r_wrist_roll_link",
            manipulator=right_gripper,
            world=world,
        )
        self.add_arm(right_arm)

    def _setup_arm_hardware_interfaces(self):
        for arm in self.arms:
            for connection in arm.active_connections:
                connection.has_hardware_interface = True

    def _setup_arm_joint_state(self):
        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [-1.712, -0.256, -1.463, -2.12, 1.766, -0.07, 0.051],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.right_arm.add_joint_state(right_arm_park)

        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.left_arm.active_connections],
                    [
                        1.712,
                        -0.264,
                        1.38,
                        -2.12,
                        16.996 + 3.14159,
                        -0.073,
                        0.0,
                    ],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.left_arm.add_joint_state(left_arm_park)

    def _setup_torso_semantic_annotations(self):

        torso = Torso.create_and_add_to_world(
            name=PrefixedName("torso", prefix=self.name.name),
            root_name="base_link",
            tip_name="head_tilt_link",
            world=self._world,
        )
        self.add_torso(torso)

    def _setup_torso_hardware_interfaces(self):
        for connection in self.torso.active_connections:
            connection.has_hardware_interface = True

    def _setup_torso_joint_state(self):
        torso_joint = [self._world.get_connection_by_name("torso_lift_joint")]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0115])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.15])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.3])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)

    def _setup_neck_semantic_annotations(self):
        # Create camera and neck
        camera = Camera.create_and_add_to_world(
            name=PrefixedName("wide_stereo_optical_frame", prefix=self.name.name),
            root_name="wide_stereo_optical_frame",
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            world=self._world,
            default_camera=True,
        )

        neck = Neck.create_and_add_to_world(
            name=PrefixedName("neck", prefix=self.name.name),
            root_name="base_link",
            tip_name="head_tilt_link",
            sensors=[camera],
            world=self._world,
        )
        self.add_neck(neck)

    def _setup_neck_hardware_interfaces(self):
        for connection in self.neck.active_connections:
            connection.has_hardware_interface = True

    def _setup_neck_joint_state(self):
        neck_joint = [self._world.get_connection_by_name("head_lift_joint")]
        ...  # Neck STATES

    def _setup_collision_rules(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "pr2.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1, violated_distance=0.0, robot=self
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    robot=self,
                    body_subset=self.left_arm.bodies_with_collision
                    + self.right_arm.bodies_with_collision,
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.2,
                    violated_distance=0.05,
                    robot=self,
                    body_subset={self._world.get_body_by_name("base_link")},
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

        self._world.collision_manager.extend_max_avoided_bodies_rules(
            [
                MaxAvoidedCollisionsOverride(
                    2, bodies={self._world.get_body_by_name("base_link")}
                ),
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("r_wrist_roll_link")
                        )
                    )
                    | set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("l_wrist_roll_link")
                        )
                    ),
                ),
            ]
        )

    def _setup_mobile_base_semantic_annotations(self):
        base = MobileBase.create_and_add_to_world(
            name=PrefixedName("base", prefix=self.name.name),
            root_name="base_link",
            world=self._world,
            main_axis=Vector3.X(),
            full_body_controlled=True,
        )
        self.add_mobile_base(base)

    def _setup_other_hardware_interfaces(self):
        pass
