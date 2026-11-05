from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import DifferentialDrive
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import UnexpectedWorldEntityCountError
from giskardpy.motion_statechart.graph_node import Goal, MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPositionStraight,
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import DegreeOfFreedomVelocityLimit


@dataclass(eq=False, repr=False)
class DifferentialDriveBaseGoal(Sequence):
    """
    A sequence that moves the robot to a goal pose using a differential drive.

    1. Orient to goal position
    2. Drive to goal position
    3. Orient to goal orientation

    The direction to the goal is an expression over the base's forward kinematics, so
    steps 1 and 2 follow the base as it drives. The goal is reached at the base's own
    height, which the drive cannot change. Step 1 turns on the spot, see
    :attr:`translation_velocity_while_turning`, and towards the goal's orientation if the
    base already stands within :attr:`threshold` of the goal position. Every step drives
    and turns as fast as the velocity limits of :attr:`diff_drive_connection` allow.

    The goal pose is held where it was, when the goal started, relative to what the drive
    stands on, the parent of :attr:`diff_drive_connection`. The goal moves no joint above
    the drive, such as an elevator cabin's.
    """

    diff_drive_connection: DifferentialDrive | None = field(kw_only=True, default=None)
    """
    Drive connection to use.

    If it is None and there is only one diff drive in the world, it will be used.
    """

    goal_pose: Pose = field(kw_only=True)
    """
    Pose to reach.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Task priority relative to other tasks.
    """

    nodes: list[MotionStatechartNode] = field(default_factory=list, init=False)

    threshold: float = field(default=0.01, kw_only=True)
    """
    Threshold when the drive goals for the base are considered achieved.
    """

    heading_threshold: float = field(default=0.02, kw_only=True)
    """
    Angle in radians within which the base has to face the goal position before it
    drives there; the smaller of this and :attr:`threshold` applies.
    """

    translation_velocity_while_turning: float = field(default=0.001, kw_only=True)
    """
    Speed in m/s the base may drive at while it turns towards the goal position, which
    keeps it turning on the spot instead of driving an arc.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        if self.diff_drive_connection is None:
            diff_drives = context.world.get_connections_by_type(DifferentialDrive)
            if len(diff_drives) == 0:
                raise UnexpectedWorldEntityCountError(
                    node=self,
                    expected_count=1,
                    actual_count=0,
                    entity_type=DifferentialDrive,
                )
            if len(diff_drives) > 1:
                raise UnexpectedWorldEntityCountError(
                    node=self,
                    expected_count=1,
                    actual_count=len(diff_drives),
                    entity_type=DifferentialDrive,
                )
            self.diff_drive_connection = diff_drives[0]
        translation_speed = self.diff_drive_connection.x_velocity.limits.upper.velocity
        rotation_speed = self.diff_drive_connection.yaw.limits.upper.velocity
        root = self.diff_drive_connection.parent
        tip = self.diff_drive_connection.child

        root_T_given_goal = context.world.transform(self.goal_pose, root)
        root_T_current = context.world.compose_forward_kinematics_expression(root, tip)
        # The drive cannot change the base's height, so the goal lies at that height
        root_T_goal = Pose(
            position=Point3(
                root_T_given_goal.x,
                root_T_given_goal.y,
                root_T_current.z,
                reference_frame=root,
            ),
            orientation=root_T_given_goal.orientation,
            reference_frame=root,
        )
        root_V_current_to_goal = root_T_goal.position - root_T_current.position
        distance_to_goal = root_V_current_to_goal.norm()
        root_V_current_to_goal.scale(1)
        root_V_z = Vector3.Z(reference_frame=root)
        # A goal within the threshold is too close for the direction to it to mean
        # anything, so the base turns to the goal's orientation instead.
        root_R_first_orientation = sm.if_less(
            distance_to_goal,
            self.threshold,
            root_T_goal.rotation_matrix,
            RotationMatrix.from_vectors(
                x=root_V_current_to_goal, z=root_V_z, reference_frame=root
            ),
        )
        root_R_first_orientation.reference_frame = root

        root_T_goal2 = Pose(
            position=root_T_goal.position,
            orientation=root_R_first_orientation.quaternion,
            reference_frame=root,
        )

        self.nodes = [
            Parallel(
                name=f"{self.name}/step1",
                nodes=[
                    CartesianOrientation(
                        name=f"{self.name}/step1/orientation",
                        root_link=root,
                        tip_link=tip,
                        goal_orientation=root_R_first_orientation,
                        reference_velocity=rotation_speed,
                        weight=self.weight,
                        threshold=min(self.threshold, self.heading_threshold),
                    ),
                    DegreeOfFreedomVelocityLimit(
                        name=f"{self.name}/step1/stay_in_place",
                        degrees_of_freedom=[self.diff_drive_connection.x_velocity],
                        max_velocity=self.translation_velocity_while_turning,
                        # Above the turn itself, so a base that lags behind its commands
                        # keeps turning on the spot instead of trading it for reversing
                        weight=DefaultWeights.WEIGHT_MAXIMUM,
                    ),
                ],
            ),
            CartesianPose(
                name=f"{self.name}/step2",
                root_link=root,
                tip_link=tip,
                goal_pose=root_T_goal2,
                reference_linear_velocity=translation_speed,
                reference_angular_velocity=rotation_speed,
                weight=self.weight,
                translation_threshold=self.threshold,
                orientation_threshold=self.threshold,
            ),
            CartesianPose(
                name=f"{self.name}/step3",
                root_link=root,
                tip_link=tip,
                goal_pose=root_T_goal,
                reference_linear_velocity=translation_speed,
                reference_angular_velocity=rotation_speed,
                weight=self.weight,
                translation_threshold=self.threshold,
                orientation_threshold=self.threshold,
            ),
        ]
        super().expand(context)


@dataclass(eq=False, repr=False)
class CartesianPoseStraight(Parallel):
    """
    Like CartesianPose, but constrains the tip link to move in a straight line towards
    the goal.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    """
    Name of the root link of the kin chain.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    Name of the tip link of the kin chain.
    """

    goal_pose: Pose = field(kw_only=True)
    """
    The goal pose.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Task priority relative to other tasks.
    """

    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_at_build, kw_only=True
    )
    """
    Describes when the goal is computed.

    See GoalBindingPolicy for more information.
    """

    nodes: list[MotionStatechartNode] = field(default_factory=list, init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        self.nodes = [
            CartesianPositionStraight(
                name=self.name + "/position",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_point=self.goal_pose.position,
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
            CartesianOrientation(
                name=self.name + "/orientation",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_orientation=self.goal_pose.rotation_matrix,
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
        ]
        super().expand(context)
